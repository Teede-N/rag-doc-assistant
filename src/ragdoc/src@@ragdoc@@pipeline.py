"""Pipeline RAG configurable : corpus -> chunks -> index -> recherche -> reranking -> réponse."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import generate as gen
from .ingest import Chunk, build_chunks, load_corpus
from .rerank import CrossEncoderReranker, LtrReranker
from .retrievers import (BM25Retriever, CharTfidfRetriever, HybridRetriever, Retriever,
                         SentenceTransformerRetriever, SpacyDenseRetriever, entropie_top)
from .text import tokenize

RACINE = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    corpus: str = str(RACINE / "corpus")
    chunking: str = "sections"            # sections | fixed | sentences
    chunking_params: dict = field(default_factory=dict)
    retriever: str = "hybrid"             # bm25 | char_tfidf | dense_spacy | dense_st | hybrid
    hybrid_composants: list = field(default_factory=lambda: ["bm25", "char_tfidf"])
    hybrid_poids: list = field(default_factory=lambda: [1.0, 0.8])
    dense_modele: str = "intfloat/multilingual-e5-small"
    k_candidats: int = 20                 # candidats passés au reranker
    k_contexte: int = 4                   # passages fournis au générateur
    reranker: str = "none"                # none | ltr | cross_encoder
    generateur: str = "auto"              # auto | extractif | claude
    seuil_abstention: float = 0.35
    exclure_injections: bool = True

    @classmethod
    def depuis_yaml(cls, chemin: str | Path) -> "Config":
        return cls(**yaml.safe_load(Path(chemin).read_text(encoding="utf-8")))


@dataclass
class Resultat:
    question: str
    reponse: gen.Reponse
    candidats: list
    confiance: float
    duree_ms: float
    injections_ecartees: int


class Pipeline:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        c = self.config
        self.docs = load_corpus(c.corpus)
        self.chunks: list[Chunk] = build_chunks(self.docs, c.chunking, **c.chunking_params)
        self.index = {ch.chunk_id: i for i, ch in enumerate(self.chunks)}
        self.retrievers: dict[str, Retriever] = {}
        self._construire_retrievers()
        self.reranker = None
        if c.reranker == "ltr":
            self.reranker = LtrReranker(self._retrievers_features(), self.chunks)
        elif c.reranker == "cross_encoder":
            self.reranker = CrossEncoderReranker()
        self.generateur = gen.choisir_generateur(c.generateur)

    # ------------------------------------------------------------------ construction
    def _fabrique(self, nom: str) -> Retriever:
        if nom == "bm25":
            return BM25Retriever(self.chunks)
        if nom == "char_tfidf":
            return CharTfidfRetriever(self.chunks)
        if nom == "dense_spacy":
            return SpacyDenseRetriever(self.chunks)
        if nom == "dense_st":
            return SentenceTransformerRetriever(self.chunks, self.config.dense_modele)
        raise ValueError(f"Retriever inconnu : {nom}")

    def _construire_retrievers(self):
        c = self.config
        noms = set(c.hybrid_composants) if c.retriever == "hybrid" else {c.retriever}
        noms |= {"bm25", "char_tfidf"}  # toujours disponibles : features du reranker et confiance
        if c.reranker == "ltr" and "dense_st" not in noms:
            noms.add("dense_spacy")       # feature « dense » du reranker appris
        for n in noms:
            self.retrievers[n] = self._fabrique(n)
        if c.retriever == "hybrid":
            self.principal = HybridRetriever(self.chunks, [self.retrievers[n] for n in c.hybrid_composants],
                                             c.hybrid_poids)
        else:
            self.principal = self.retrievers[c.retriever]

    def _retrievers_features(self) -> dict[str, Retriever]:
        feats = {"bm25": self.retrievers["bm25"], "char_tfidf": self.retrievers["char_tfidf"]}
        dense = self.retrievers.get("dense_st") or self.retrievers.get("dense_spacy")
        if dense:
            feats["dense"] = dense
        return feats

    # ------------------------------------------------------------------ recherche
    def rechercher(self, question: str, k: int | None = None):
        k = k or self.config.k_contexte
        hits = self.principal.search(question, self.config.k_candidats)
        if self.reranker:
            hits = self.reranker.rerank(question, hits)
        return hits[:k]

    def confiance(self, question: str, hits) -> float:
        """Score de confiance [0, 1] pour décider de l'abstention.

        Un moteur lexical ne sait pas dire « je ne sais pas » : un seul mot commun suffit à produire
        un résultat. On combine donc trois signaux calibrés sur le corpus :
        (a) le meilleur score BM25, saturé (une vraie réponse partage plusieurs termes rares) ;
        (b) la meilleure similarité caractères (robuste aux fautes) ;
        (c) la marge entre les deux meilleurs scores (une réponse nette domine ses voisines).
        Le seuil est ensuite choisi par le banc d'évaluation (voir evaluate.calibrer_seuil).
        """
        if not hits:
            return 0.0
        s_bm25 = self.retrievers["bm25"].scores(question)
        s_char = self.retrievers["char_tfidf"].scores(question)
        tri = np.sort(s_bm25)[::-1]
        marge = float((tri[0] - tri[1]) / (tri[0] + 1e-9)) if len(tri) > 1 and tri[0] > 0 else 0.0
        return float(np.clip(0.5 * min(1.0, tri[0] / 14.0) + 0.3 * min(1.0, float(s_char.max()) / 0.35)
                             + 0.2 * marge, 0, 1))

    # ------------------------------------------------------------------ réponse
    def repondre(self, question: str) -> Resultat:
        t0 = time.perf_counter()
        hits = self.rechercher(question)
        injections = 0
        if self.config.exclure_injections:
            propres = [h for h in hits if not h.chunk.injection_suspecte]
            injections = len(hits) - len(propres)
            hits = propres
        conf = self.confiance(question, hits)
        if conf < self.config.seuil_abstention or not hits:
            rep = gen.Reponse(gen.MESSAGE_ABSTENTION, [], True, conf, self.generateur.nom,
                              gen.verifier_ancrage("", []))
        else:
            texte, cites = self.generateur.generer(question, hits)
            abst = texte.strip() == gen.MESSAGE_ABSTENTION
            rep = gen.Reponse(texte, cites, abst, conf, self.generateur.nom,
                              gen.verifier_ancrage(texte, [h.chunk.text for h in hits]))
        return Resultat(question, rep, hits, conf, round((time.perf_counter() - t0) * 1000, 1), injections)
