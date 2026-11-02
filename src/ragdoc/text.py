"""Normalisation et tokenisation du français (sans dépendance lourde)."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import snowballstemmer

_STEMMER = snowballstemmer.stemmer("french")

# Mots vides français courants (liste volontairement courte : les mots porteurs de sens restent)
STOPWORDS = set("""
le la les l un une des du de d et ou où à a au aux en dans sur sous par pour avec sans ce cet cette ces
se sa son ses leur leurs mon ma mes ton ta tes notre nos votre vos est sont été être avoir ai as ont
il elle ils elles on nous vous je tu y ne pas plus que qui quoi dont quel quelle quels quelles
comment combien quand est-ce c ça cela celui celle ceux qu si mais donc or ni car tout tous toute
toutes très peut peuvent doit doivent faut fait faire
""".split())

_RE_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize(s: str) -> str:
    """Minuscules, accents retirés, apostrophes et espaces insécables neutralisés."""
    s = strip_accents(s.lower()).replace("’", "'").replace(" ", " ").replace(" ", " ")
    return s.replace("'", " ")


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    return _STEMMER.stemWord(word)


def tokenize(s: str, remove_stopwords: bool = True, do_stem: bool = True) -> list[str]:
    toks = _RE_TOKEN.findall(normalize(s))
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]
    if do_stem:
        toks = [stem(t) for t in toks]
    return toks


_RE_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý«(0-9])")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _RE_SENT.split(text.replace("\n", " "))]
    return [p for p in parts if p]


def char_ngrams(s: str, n_min: int = 3, n_max: int = 5) -> list[str]:
    """N-grammes de caractères (robustes aux fautes de frappe)."""
    s = " " + " ".join(normalize(s).split()) + " "
    out = []
    for n in range(n_min, n_max + 1):
        out.extend(s[i:i + n] for i in range(len(s) - n + 1))
    return out
