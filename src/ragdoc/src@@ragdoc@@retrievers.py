"""Méthodes de recherche : lexicale (BM25), caractères (TF-IDF n-grammes), dense, hybride (RRF).

Toutes exposent la même interface : `search(query, k) -> list[Hit]` avec un score, ce qui permet
de les comparer dans le banc d'évaluation et de les combiner.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from .ingest import Chunk
from .text import STOPWORDS, char_ngrams, normalize, tokenize


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rang: int
    source: str
    details: dict | None = None


class Retriever:
    nom = "base"

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks

    def scores(self, query: str) -> np.ndarray:  # un score par chunk, plus haut = plus pertinent
        raise NotImplementedError

    def search(self, query: str, k: int = 5) -> list[Hit]:
        s = self.scores(query)
        ordre = np.argsort(-s)[:k]
        return [Hit(self.chunks[i], float(s[i]), r + 1, self.nom) for r, i in enumerate(ordre)]


class BM25Retriever(Retriever):
    """Lexical : mots racinisés, sans mots vides. Précis sur les termes exacts, sensible aux fautes."""
    nom = "bm25"

    def __init__(self, chunks, k1: float = 1.5, b: float = 0.75):
        super().__init__(chunks)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks], k1=k1, b=b)

    def scores(self, query):
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=float)


class CharTfidfRetriever(Retriever):
    """TF-IDF sur n-grammes de caractères : tolère les fautes de frappe et les variantes."""
    nom = "char_tfidf"

    def __init__(self, chunks):
        super().__init__(chunks)
        self.vec = TfidfVectorizer(analyzer=char_ngrams, sublinear_tf=True, min_df=1)
        self.matrix = self.vec.fit_transform([c.text for c in chunks])

    def scores(self, query):
        q = self.vec.transform([query])
        return np.asarray((self.matrix @ q.T).todense()).ravel()


class SpacyDenseRetriever(Retriever):
    """Dense « léger » : moyenne des vecteurs de mots français de spaCy (fr_core_news_md, 300 d).

    Capte des synonymes et des reformulations, mais ignore l'ordre des mots : c'est une base de
    comparaison, pas un modèle de phrases. Voir SentenceTransformerRetriever pour un vrai encodeur.
    """
    nom = "dense_spacy"

    def __init__(self, chunks, modele: str = "fr_core_news_md"):
        super().__init__(chunks)
        import spacy
        self.nlp = spacy.load(modele, exclude=["parser", "ner", "lemmatizer", "attribute_ruler", "tok2vec"])
        self.matrix = self._encode([c.text for c in chunks])

    def _encode(self, textes: list[str]) -> np.ndarray:
        out = []
        for t in textes:
            doc = self.nlp.make_doc(t)
            vecs = [tok.vector for tok in doc if tok.has_vector and normalize(tok.text) not in STOPWORDS
                    and not tok.is_punct and not tok.is_space]
            v = np.mean(vecs, axis=0) if vecs else np.zeros(self.nlp.vocab.vectors_length)
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.vstack(out)

    def scores(self, query):
        return self.matrix @ self._encode([query])[0]


class SentenceTransformerRetriever(Retriever):
    """Dense « fort » : encodeur de phrases (par défaut multilingual-e5-small). Nécessite le téléchargement
    du modèle depuis Hugging Face ; activé dans la configuration quand il est disponible."""
    nom = "dense_st"

    def __init__(self, chunks, modele: str = "intfloat/multilingual-e5-small"):
        super().__init__(chunks)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(modele)
        self.prefixe_q, self.prefixe_p = ("query: ", "passage: ") if "e5" in modele else ("", "")
        self.matrix = self.model.encode([self.prefixe_p + c.text for c in chunks], normalize_embeddings=True)

    def scores(self, query):
        q = self.model.encode([self.prefixe_q + query], normalize_embeddings=True)[0]
        return self.matrix @ q


class HybridRetriever(Retriever):
    """Fusion de plusieurs retrievers par Reciprocal Rank Fusion (RRF) : robuste, sans calibration
    des scores entre méthodes. score = Σ poids_m / (k_rrf + rang_m)."""
    nom = "hybrid"

    def __init__(self, chunks, retrievers: list[Retriever], poids: list[float] | None = None, k_rrf: int = 60):
        super().__init__(chunks)
        self.retrievers = retrievers
        self.poids = poids or [1.0] * len(retrievers)
        self.k_rrf = k_rrf
        self.nom = "hybrid(" + "+".join(r.nom for r in retrievers) + ")"

    def scores(self, query):
        total = np.zeros(len(self.chunks))
        for r, w in zip(self.retrievers, self.poids):
            s = r.scores(query)
            rangs = np.empty(len(s), dtype=int)
            rangs[np.argsort(-s)] = np.arange(1, len(s) + 1)
            total += w / (self.k_rrf + rangs)
        return total


def normaliser_scores(s: np.ndarray) -> np.ndarray:
    """Min-max sur la distribution d'une requête, pour les features du reranker."""
    lo, hi = float(s.min()), float(s.max())
    return (s - lo) / (hi - lo) if hi > lo else np.zeros_like(s)


def entropie_top(scores: np.ndarray, k: int = 5) -> float:
    """Mesure de « netteté » du résultat : faible quand un chunk domine, élevée quand tout se vaut."""
    top = np.sort(scores)[-k:]
    top = top - top.min() + 1e-9
    p = top / top.sum()
    return float(-(p * np.log(p)).sum() / math.log(k))
