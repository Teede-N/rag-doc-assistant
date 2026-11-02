"""Chargement du corpus Markdown et stratégies de découpage (chunking)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .text import split_sentences, tokenize

# Motifs d'injection de consignes dans un document (traités comme des données, jamais exécutés)
RE_INJECTION = re.compile(
    r"(ignore[sz]?\s+(toutes\s+)?(les\s+)?instructions|instruction\s+syst[eè]me|r[ée]ponds\s+uniquement"
    r"|assistant\s*:|system\s+prompt|disregard\s+(all\s+)?(previous|prior)\s+instructions)", re.I)


@dataclass
class Document:
    doc_id: str
    titre: str
    service: str
    version: str
    date: str
    sections: list[tuple[str, str]]   # (titre de section, texte)
    chemin: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_titre: str
    section: str
    text: str
    position: int
    injection_suspecte: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def ref(self) -> str:
        """Référence courte utilisée pour les citations et l'évaluation : « 03#Entretien périodique »."""
        return f"{self.doc_id}#{self.section}"

    @property
    def refs(self) -> list[str]:
        """Toutes les sections couvertes (un chunk à taille fixe peut en chevaucher plusieurs)."""
        return [f"{self.doc_id}#{s}" for s in self.meta.get("sections", [self.section])]


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    if raw.startswith("---"):
        _, fm, body = raw.split("---", 2)
        return yaml.safe_load(fm) or {}, body
    return {}, raw


def load_corpus(dossier: Path) -> list[Document]:
    docs = []
    for f in sorted(Path(dossier).glob("*.md")):
        meta, body = _parse_frontmatter(f.read_text(encoding="utf-8"))
        doc_id = f.stem.split("_")[0]
        sections: list[tuple[str, str]] = []
        titre_section, tampon = "", []
        for ligne in body.splitlines():
            if ligne.startswith("## "):
                if tampon and "".join(tampon).strip():
                    sections.append((titre_section, "\n".join(tampon).strip()))
                titre_section, tampon = ligne[3:].strip(), []
            elif ligne.startswith("# "):
                continue
            else:
                tampon.append(ligne)
        if tampon and "".join(tampon).strip():
            sections.append((titre_section, "\n".join(tampon).strip()))
        docs.append(Document(doc_id, str(meta.get("titre", f.stem)), str(meta.get("service", "")),
                             str(meta.get("version", "")), str(meta.get("date", "")), sections, str(f)))
    return docs


def _table_to_text(text: str) -> str:
    """Transforme les tableaux Markdown en phrases « colonne : valeur » lisibles par le moteur."""
    lignes, entete, out = text.splitlines(), None, []
    for l in lignes:
        if l.startswith("|"):
            cells = [c.strip() for c in l.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if entete is None:
                entete = cells
                continue
            out.append(" ; ".join(f"{h} : {v}" for h, v in zip(entete, cells)) + ".")
        else:
            entete = None
            out.append(l)
    return "\n".join(out)


def chunk_sections(docs: list[Document]) -> list[Chunk]:
    """Un chunk = une section (titre inclus). Grain naturel pour des procédures."""
    chunks = []
    for d in docs:
        for i, (titre, texte) in enumerate(d.sections):
            t = f"{d.titre} – {titre}\n{_table_to_text(texte)}"
            chunks.append(Chunk(f"{d.doc_id}-s{i}", d.doc_id, d.titre, titre, t, i,
                                bool(RE_INJECTION.search(texte))))
    return chunks


def chunk_fixed(docs: list[Document], taille: int = 120, recouvrement: int = 30) -> list[Chunk]:
    """Fenêtres de `taille` mots avec recouvrement, sans tenir compte des sections."""
    chunks = []
    for d in docs:
        mots, bornes = [], []
        for titre, texte in d.sections:
            ws = _table_to_text(texte).split()
            bornes.append((len(mots), len(mots) + len(ws), titre))
            mots.extend(ws)
        pas, i, k = max(1, taille - recouvrement), 0, 0
        while i < len(mots):
            fen = mots[i:i + taille]
            couvertes = [t for a, b, t in bornes if a < i + len(fen) and b > i]
            section = couvertes[0] if couvertes else ""
            texte = f"{d.titre}\n" + " ".join(fen)
            chunks.append(Chunk(f"{d.doc_id}-f{k}", d.doc_id, d.titre, section, texte, k,
                                bool(RE_INJECTION.search(texte)), {"sections": couvertes}))
            i, k = i + pas, k + 1
    return chunks


def chunk_sentences(docs: list[Document], fenetre: int = 3, pas: int = 2) -> list[Chunk]:
    """Fenêtres glissantes de `fenetre` phrases à l'intérieur de chaque section."""
    chunks = []
    for d in docs:
        k = 0
        for titre, texte in d.sections:
            phrases = split_sentences(_table_to_text(texte))
            for i in range(0, max(1, len(phrases) - fenetre + 1), pas):
                t = f"{d.titre} – {titre}\n" + " ".join(phrases[i:i + fenetre])
                chunks.append(Chunk(f"{d.doc_id}-p{k}", d.doc_id, d.titre, titre, t, k,
                                    bool(RE_INJECTION.search(t))))
                k += 1
    return chunks


CHUNKERS = {"sections": chunk_sections, "fixed": chunk_fixed, "sentences": chunk_sentences}


def build_chunks(docs: list[Document], strategie: str = "sections", **kw) -> list[Chunk]:
    return CHUNKERS[strategie](docs, **kw)


def stats(chunks: list[Chunk]) -> dict:
    tailles = [len(tokenize(c.text, remove_stopwords=False, do_stem=False)) for c in chunks]
    return {"nb_chunks": len(chunks), "mots_moyen": round(sum(tailles) / len(tailles), 1),
            "mots_max": max(tailles), "injections_detectees": sum(c.injection_suspecte for c in chunks)}
