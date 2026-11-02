"""Reranking des candidats.

- `LtrReranker` : petit modèle de learning-to-rank (régression logistique) entraîné sur des
  features interprétables (scores des retrievers, couverture des termes, correspondance du titre).
  Entraîné et évalué en validation croisée par question dans le banc d'évaluation.
- `CrossEncoderReranker` : cross-encoder Hugging Face (optionnel, plus coûteux, plus précis).
"""
from __future__ import annotations

import numpy as np

from .ingest import Chunk
from .retrievers import Hit, Retriever, normaliser_scores
from .text import tokenize

FEATURES = ["bm25", "char_tfidf", "dense", "couverture_termes", "titre_match", "longueur", "rang_min"]


def features_candidats(query: str, candidats: list[Chunk], retrievers: dict[str, Retriever],
                       index: dict[str, int]) -> np.ndarray:
    """Matrice (n_candidats × n_features) pour une requête."""
    q_toks = set(tokenize(query))
    scores = {nom: normaliser_scores(r.scores(query)) for nom, r in retrievers.items()}
    rangs = {}
    for nom, r in retrievers.items():
        s = r.scores(query)
        rk = np.empty(len(s), dtype=int)
        rk[np.argsort(-s)] = np.arange(1, len(s) + 1)
        rangs[nom] = rk
    X = []
    for c in candidats:
        i = index[c.chunk_id]
        c_toks = set(tokenize(c.text))
        titre_toks = set(tokenize(c.section + " " + c.doc_titre))
        X.append([
            scores.get("bm25", np.zeros(1))[i] if "bm25" in scores else 0.0,
            scores.get("char_tfidf", np.zeros(1))[i] if "char_tfidf" in scores else 0.0,
            scores.get("dense", np.zeros(1))[i] if "dense" in scores else 0.0,
            len(q_toks & c_toks) / max(1, len(q_toks)),
            len(q_toks & titre_toks) / max(1, len(q_toks)),
            min(1.0, len(c_toks) / 150),
            1.0 / min(rk[i] for rk in rangs.values()),
        ])
    return np.asarray(X, dtype=float)


class LtrReranker:
    nom = "ltr"

    def __init__(self, retrievers: dict[str, Retriever], chunks: list[Chunk]):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.retrievers = retrievers
        self.index = {c.chunk_id: i for i, c in enumerate(chunks)}
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))
        self.entraine = False

    def fit(self, exemples: list[tuple[str, list[Chunk], list[int]]]) -> "LtrReranker":
        """exemples : (question, candidats, labels 0/1 : le candidat contient-il la réponse ?)"""
        X = np.vstack([features_candidats(q, c, self.retrievers, self.index) for q, c, _ in exemples])
        y = np.concatenate([np.asarray(l) for _, _, l in exemples])
        self.model.fit(X, y)
        self.entraine = True
        return self

    def coefficients(self) -> dict[str, float]:
        lr = self.model.named_steps["logisticregression"]
        return dict(zip(FEATURES, [round(float(c), 3) for c in lr.coef_[0]]))

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not self.entraine or not hits:
            return hits
        X = features_candidats(query, [h.chunk for h in hits], self.retrievers, self.index)
        p = self.model.predict_proba(X)[:, 1]
        ordre = np.argsort(-p)
        return [Hit(hits[i].chunk, float(p[i]), r + 1, "ltr") for r, i in enumerate(ordre)]


class CrossEncoderReranker:
    """Reranker neuronal (optionnel) : modèle multilingue par défaut, téléchargé depuis Hugging Face."""
    nom = "cross_encoder"

    def __init__(self, modele: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(modele)

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        s = self.model.predict([(query, h.chunk.text) for h in hits])
        ordre = np.argsort(-s)
        return [Hit(hits[i].chunk, float(s[i]), r + 1, "cross_encoder") for r, i in enumerate(ordre)]
