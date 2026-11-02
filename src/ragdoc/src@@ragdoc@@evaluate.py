"""Banc d'évaluation : grille de configurations, métriques de recherche et de réponse, robustesse.

Usage : PYTHONPATH=src python -m ragdoc.evaluate [--rapide] [--claude]
Sorties : reports/evaluation.md, reports/evaluation.json, reports/*.png
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from . import generate as gen
from .ingest import Chunk
from .pipeline import RACINE, Config, Pipeline
from .rerank import LtrReranker
from .retrievers import Hit

QUESTIONS = RACINE / "eval" / "questions.jsonl"
REPORTS = RACINE / "reports"


def charger_questions() -> list[dict]:
    return [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]


# ----------------------------------------------------------------------------- métriques recherche
def metriques_recherche(hits: list[Hit], gold: list[str], k: int) -> dict:
    refs = [set(h.chunk.refs) for h in hits[:k]]
    if not gold:
        return {}
    trouves = [g for g in gold if any(g in r for r in refs)]
    rang_premier = next((i + 1 for i, r in enumerate(refs) if r & set(gold)), None)
    # Chaque section attendue ne rapporte un gain qu'une fois (plusieurs chunks peuvent couvrir la même section).
    dcg, deja_vus = 0.0, set()
    for i, r in enumerate(refs):
        nouveaux = (r & set(gold)) - deja_vus
        if nouveaux:
            dcg += 1 / math.log2(i + 2)
            deja_vus |= nouveaux
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return {"recall": len(trouves) / len(gold), "hit": float(bool(trouves)),
            "mrr": 1 / rang_premier if rang_premier else 0.0, "ndcg": dcg / idcg if idcg else 0.0}


def moyenne(liste: list[dict], cle: str) -> float:
    vals = [d[cle] for d in liste if cle in d]
    return round(float(np.mean(vals)), 3) if vals else float("nan")


# ----------------------------------------------------------------------------- reranker en validation croisée
def rerank_cv(pipe: Pipeline, questions: list[dict], k_cand: int, n_folds: int = 5) -> dict[str, list[Hit]]:
    """Entraîne le reranker LTR sur n−1 plis de questions et reclasse les questions du pli restant."""
    answerables = [q for q in questions if q["gold"]]
    resultats: dict[str, list[Hit]] = {}
    candidats = {q["id"]: pipe.principal.search(q["question"], k_cand) for q in questions}
    for f in range(n_folds):
        test_ids = {q["id"] for i, q in enumerate(answerables) if i % n_folds == f}
        exemples = []
        for q in answerables:
            if q["id"] in test_ids:
                continue
            cands = [h.chunk for h in candidats[q["id"]]]
            exemples.append((q["question"], cands, [int(bool(set(c.refs) & set(q["gold"]))) for c in cands]))
        rr = LtrReranker(pipe._retrievers_features(), pipe.chunks).fit(exemples)
        for q in questions:
            if q["id"] in test_ids or not q["gold"]:
                resultats[q["id"]] = rr.rerank(q["question"], candidats[q["id"]])
    return resultats


# ----------------------------------------------------------------------------- évaluation d'une configuration
def evaluer_config(nom: str, cfg: Config, questions: list[dict], k: int = 5, ltr: bool = False) -> dict:
    t0 = time.perf_counter()
    pipe = Pipeline(cfg)
    t_build = time.perf_counter() - t0
    hits_par_q: dict[str, list[Hit]] = {}
    if ltr:
        hits_par_q = rerank_cv(pipe, questions, cfg.k_candidats)
    lignes, durees = [], []
    for q in questions:
        t1 = time.perf_counter()
        hits = hits_par_q.get(q["id"]) or pipe.principal.search(q["question"], cfg.k_candidats)
        durees.append((time.perf_counter() - t1) * 1000)
        m = metriques_recherche(hits, q["gold"], k)
        m.update({"id": q["id"], "type": q["type"], "confiance": pipe.confiance(q["question"], hits[:cfg.k_contexte])})
        lignes.append(m)
    par_type = {}
    for t in sorted({l["type"] for l in lignes}):
        sub = [l for l in lignes if l["type"] == t]
        par_type[t] = {"n": len(sub), "recall": moyenne(sub, "recall"), "mrr": moyenne(sub, "mrr")}
    return {"config": nom, "chunking": cfg.chunking, "retriever": cfg.retriever + ("+ltr" if ltr else ""),
            "nb_chunks": len(pipe.chunks), "build_s": round(t_build, 2), "latence_ms": round(float(np.median(durees)), 1),
            f"recall@{k}": moyenne(lignes, "recall"), "hit@1": round(float(np.mean([l["mrr"] == 1.0 for l in lignes if "mrr" in l])), 3),
            "mrr": moyenne(lignes, "mrr"), f"ndcg@{k}": moyenne(lignes, "ndcg"), "par_type": par_type,
            "_lignes": lignes, "_pipe": pipe, "_hits": hits_par_q}


# ----------------------------------------------------------------------------- abstention
def calibrer_seuil(lignes: list[dict], questions: list[dict]) -> dict:
    """Choisit le seuil de confiance qui maximise l'exactitude globale (répondre quand il faut,
    s'abstenir quand il faut), estimé en validation croisée à 2 plis, puis fixé sur l'ensemble."""
    gold = {q["id"]: bool(q["gold"]) for q in questions}
    conf = {l["id"]: l["confiance"] for l in lignes}
    ids = sorted(conf)
    seuils = np.round(np.arange(0.05, 0.95, 0.01), 2)

    def exactitude(sel, s):
        return np.mean([(conf[i] >= s) == gold[i] for i in sel])

    plis = [ids[0::2], ids[1::2]]
    cv = []
    for a, b in [(plis[0], plis[1]), (plis[1], plis[0])]:
        s_opt = max(seuils, key=lambda s: exactitude(a, s))
        cv.append(exactitude(b, s_opt))
    s_final = float(max(seuils, key=lambda s: exactitude(ids, s)))
    courbe = [{"seuil": float(s), "exactitude": round(float(exactitude(ids, s)), 3)} for s in seuils]
    sans = [i for i in ids if not gold[i]]
    avec = [i for i in ids if gold[i]]
    return {"seuil": s_final, "exactitude_cv": round(float(np.mean(cv)), 3),
            "exactitude": round(float(exactitude(ids, s_final)), 3),
            "abstention_correcte_sans_reponse": round(float(np.mean([conf[i] < s_final for i in sans])), 3),
            "fausses_abstentions": round(float(np.mean([conf[i] < s_final for i in avec])), 3),
            "courbe": courbe}


# ----------------------------------------------------------------------------- réponses finales
def evaluer_reponses(pipe: Pipeline, questions: list[dict], hits_cv: dict[str, list[Hit]] | None = None) -> dict:
    lignes = []
    for q in questions:
        if hits_cv and q["id"] in hits_cv:
            # on rejoue repondre() avec les hits reclassés en validation croisée
            hits = hits_cv[q["id"]][:pipe.config.k_contexte]
            if pipe.config.exclure_injections:
                hits = [h for h in hits if not h.chunk.injection_suspecte]
            conf = pipe.confiance(q["question"], hits)
            if conf < pipe.config.seuil_abstention or not hits:
                texte, cites = gen.MESSAGE_ABSTENTION, []
            else:
                texte, cites = pipe.generateur.generer(q["question"], hits)
            rep = gen.Reponse(texte, cites, texte == gen.MESSAGE_ABSTENTION, conf, pipe.generateur.nom,
                              gen.verifier_ancrage(texte, [h.chunk.text for h in hits]))
        else:
            rep = pipe.repondre(q["question"]).reponse
        t = rep.texte.lower()
        mots_ok = all(any(alt.lower() in t for alt in kw.split("|")) for kw in q["expected_keywords"])
        attendu_abst = not q["gold"]
        correct = rep.abstention if attendu_abst else (mots_ok and not rep.abstention)
        lignes.append({"id": q["id"], "type": q["type"], "abstention": rep.abstention, "mots_ok": mots_ok,
                       "correct": correct, "fuite_injection": "0000" in rep.texte,
                       "phrases_ancrees": rep.ancrage["phrases_ancrees"],
                       "nombres_non_ancres": rep.ancrage["nombres_non_ancres"], "reponse": rep.texte[:300],
                       "citations": [h.chunk.ref for h in rep.citations]})
    par_type = {}
    for ty in sorted({l["type"] for l in lignes}):
        sub = [l for l in lignes if l["type"] == ty]
        par_type[ty] = {"n": len(sub), "exactitude": round(float(np.mean([l["correct"] for l in sub])), 3)}
    return {"generateur": pipe.generateur.nom, "exactitude_globale": round(float(np.mean([l["correct"] for l in lignes])), 3),
            "par_type": par_type, "fuites_injection": int(sum(l["fuite_injection"] for l in lignes)),
            "phrases_ancrees": round(float(np.mean([l["phrases_ancrees"] for l in lignes])), 3),
            "reponses_avec_nombre_non_ancre": int(sum(bool(l["nombres_non_ancres"]) for l in lignes)),
            "lignes": lignes}


# ----------------------------------------------------------------------------- rapport
def tableau(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    out = ["| " + " | ".join(t for _, t in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c, _ in cols) + " |")
    return "\n".join(out)


def tracer(resultats: list[dict], calib: dict, k: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    REPORTS.mkdir(exist_ok=True)
    sec = [r for r in resultats if r["chunking"] == "sections"]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=130)
    noms = [r["retriever"] for r in sec]
    x = np.arange(len(noms))
    ax.bar(x - 0.2, [r[f"recall@{k}"] for r in sec], 0.4, label=f"Recall@{k}", color="#1d4ed8")
    ax.bar(x + 0.2, [r["mrr"] for r in sec], 0.4, label="MRR", color="#d97706")
    ax.set_xticks(x, noms, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Qualité de la recherche par méthode (découpage par sections)")
    ax.legend(frameon=False, loc="lower left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "recherche_par_methode.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=130)
    ax.plot([c["seuil"] for c in calib["courbe"]], [c["exactitude"] for c in calib["courbe"]], color="#1d4ed8")
    ax.axvline(calib["seuil"], color="#c62f3a", ls="--", label=f"seuil retenu = {calib['seuil']}")
    ax.set(xlabel="Seuil de confiance", ylabel="Exactitude (répondre / s'abstenir)",
           title="Calibration de l'abstention")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "calibration_abstention.png")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapide", action="store_true", help="n'évalue que le découpage par sections")
    ap.add_argument("--claude", action="store_true", help="évalue aussi les réponses générées par Claude")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args(argv)
    questions = charger_questions()
    k = args.k
    chunkings = ["sections"] if args.rapide else ["sections", "fixed", "sentences"]
    variantes = [("bm25", {}), ("char_tfidf", {}), ("dense_spacy", {}),
                 ("hybrid", {}),  # bm25 + char_tfidf (RRF)
                 ("hybrid+dense", {"hybrid_composants": ["bm25", "char_tfidf", "dense_spacy"], "hybrid_poids": [1.0, 0.8, 0.7]})]
    resultats = []
    for ch in chunkings:
        for nom, extra in variantes:
            cfg = Config(chunking=ch, retriever="hybrid" if nom.startswith("hybrid") else nom, generateur="extractif", **extra)
            r = evaluer_config(f"{ch}/{nom}", cfg, questions, k)
            r["retriever"] = nom
            resultats.append(r)
            print(f"{ch:9s} {nom:12s} recall@{k}={resultats[-1][f'recall@{k}']:.3f} mrr={resultats[-1]['mrr']:.3f}")
        cfg = Config(chunking=ch, retriever="hybrid", reranker="ltr", generateur="extractif")
        resultats.append(evaluer_config(f"{ch}/hybrid+ltr", cfg, questions, k, ltr=True))
        print(f"{ch:9s} {'hybrid+ltr':12s} recall@{k}={resultats[-1][f'recall@{k}']:.3f} mrr={resultats[-1]['mrr']:.3f}")

    meilleur = max(resultats, key=lambda r: (r["mrr"], r[f"recall@{k}"]))
    calib = calibrer_seuil(meilleur["_lignes"], questions)
    pipe: Pipeline = meilleur["_pipe"]
    pipe.config.seuil_abstention = calib["seuil"]
    reponses = {"extractif": evaluer_reponses(pipe, questions, meilleur["_hits"])}
    if args.claude:
        pipe.generateur = gen.ClaudeGenerator()
        reponses["claude"] = evaluer_reponses(pipe, questions, meilleur["_hits"])
    tracer(resultats, calib, k)

    # ---- rapport Markdown
    lignes_md = ["# Rapport d'évaluation", "",
                 f"Jeu d'évaluation : **{len(questions)} questions** ({sum(1 for q in questions if q['gold'])} avec réponse, "
                 f"{sum(1 for q in questions if not q['gold'])} sans réponse). Métriques de recherche calculées sur les questions avec réponse, "
                 f"top-{k}. Modèle dense évalué ici : vecteurs de mots spaCy `fr_core_news_md` (le modèle `dense_st` "
                 "multilingual-e5 n'a pas pu être téléchargé dans l'environnement d'évaluation).", "",
                 "## 1. Recherche : découpage × méthode", "",
                 tableau(resultats, [("config", "Configuration"), ("nb_chunks", "Chunks"), (f"recall@{k}", f"Recall@{k}"),
                                     ("hit@1", "Hit@1"), ("mrr", "MRR"), (f"ndcg@{k}", f"nDCG@{k}"), ("latence_ms", "Latence (ms)")]),
                 "", f"Meilleure configuration : **{meilleur['config']}** (MRR {meilleur['mrr']}).", "",
                 "![Recherche par méthode](recherche_par_methode.png)", "",
                 "## 2. Robustesse de la recherche par type de question (meilleure configuration)", "",
                 tableau([{"type": t, **v} for t, v in meilleur["par_type"].items() if t != "sans_reponse"],
                         [("type", "Type"), ("n", "N"), ("recall", f"Recall@{k}"), ("mrr", "MRR")]), ""]
    sec = {r["retriever"]: r for r in resultats if r["chunking"] == "sections"}
    if "bm25" in sec and "char_tfidf" in sec:
        lignes_md += ["Lecture : sur les questions **avec fautes de frappe**, BM25 obtient un MRR de "
                      f"{sec['bm25']['par_type'].get('typo', {}).get('mrr', 'n/a')} contre "
                      f"{sec['char_tfidf']['par_type'].get('typo', {}).get('mrr', 'n/a')} pour le TF-IDF caractères ; "
                      f"l'hybride lexical obtient {sec['hybrid']['par_type'].get('typo', {}).get('mrr', 'n/a')}. "
                      "Sur les **reformulations** en langage courant : BM25 "
                      f"{sec['bm25']['par_type'].get('reformulation', {}).get('mrr', 'n/a')}, dense spaCy "
                      f"{sec['dense_spacy']['par_type'].get('reformulation', {}).get('mrr', 'n/a')}, hybride + dense "
                      f"{sec['hybrid+dense']['par_type'].get('reformulation', {}).get('mrr', 'n/a')} : c'est là qu'un vrai "
                      "encodeur de phrases (e5) apporterait le plus.", ""]
    lignes_md += ["## 3. Abstention : savoir dire « je ne sais pas »", "",
                  f"- Seuil de confiance retenu : **{calib['seuil']}** (exactitude estimée en validation croisée : {calib['exactitude_cv']}).",
                  f"- Questions sans réponse correctement refusées : **{calib['abstention_correcte_sans_reponse']:.0%}**.",
                  f"- Questions avec réponse refusées à tort : **{calib['fausses_abstentions']:.0%}**.", "",
                  "![Calibration](calibration_abstention.png)", "",
                  "## 4. Réponses finales", ""]
    for nom, r in reponses.items():
        lignes_md += [f"### Générateur `{nom}`", "",
                      f"- Exactitude globale (bonne réponse ou abstention justifiée) : **{r['exactitude_globale']:.0%}**",
                      f"- Fuites d'injection (réponse contenant le code piégé) : **{r['fuites_injection']}**",
                      f"- Phrases ancrées dans les passages cités : **{r['phrases_ancrees']:.0%}** ; réponses avec un nombre non ancré : {r['reponses_avec_nombre_non_ancre']}", "",
                      tableau([{"type": t, **v} for t, v in r["par_type"].items()], [("type", "Type"), ("n", "N"), ("exactitude", "Exactitude")]), ""]
        erreurs = [l for l in r["lignes"] if not l["correct"]]
        if erreurs:
            lignes_md += ["<details><summary>Erreurs (" + str(len(erreurs)) + ")</summary>", ""]
            lignes_md += [f"- **{l['id']}** ({l['type']}) : {l['reponse'][:160]}… — citations {l['citations']}" for l in erreurs]
            lignes_md += ["", "</details>", ""]
    if any(r["retriever"].endswith("+ltr") for r in resultats):
        rr = LtrReranker(pipe._retrievers_features(), pipe.chunks)
        ex = []
        for q in questions:
            if q["gold"]:
                cands = [h.chunk for h in pipe.principal.search(q["question"], pipe.config.k_candidats)]
                ex.append((q["question"], cands, [int(bool(set(c.refs) & set(q["gold"]))) for c in cands]))
        rr.fit(ex)
        lignes_md += ["## 5. Reranker appris (LTR) : poids des features", "",
                      "Régression logistique sur features normalisées (entraînée ici sur toutes les questions, "
                      "après évaluation en validation croisée à 5 plis dans le tableau 1).", "",
                      tableau([{"f": f, "w": w} for f, w in rr.coefficients().items()], [("f", "Feature"), ("w", "Coefficient")]), ""]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "evaluation.md").write_text("\n".join(lignes_md) + "\n", encoding="utf-8")
    export = {"questions": len(questions), "k": k, "resultats": [{k2: v for k2, v in r.items() if not k2.startswith("_")} for r in resultats],
              "meilleure_config": meilleur["config"], "calibration": {k2: v for k2, v in calib.items() if k2 != "courbe"},
              "reponses": {n: {k2: v for k2, v in r.items() if k2 != "lignes"} for n, r in reponses.items()}}
    (REPORTS / "evaluation.json").write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nRapport : reports/evaluation.md")


if __name__ == "__main__":
    main()
