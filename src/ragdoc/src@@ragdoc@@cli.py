"""Ligne de commande.

  PYTHONPATH=src python -m ragdoc.cli ask "Quelle est la durée maximale de conduite journalière ?"
  PYTHONPATH=src python -m ragdoc.cli ask -c config/default.yaml --explain "..."
  PYTHONPATH=src python -m ragdoc.cli search "sangles" -k 3
  PYTHONPATH=src python -m ragdoc.cli stats
"""
from __future__ import annotations

import argparse
import json

from .ingest import stats
from .pipeline import Config, Pipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ragdoc")
    ap.add_argument("-c", "--config", help="fichier YAML de configuration")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="poser une question")
    a.add_argument("question")
    a.add_argument("--explain", action="store_true", help="afficher les passages, scores et l'ancrage")
    a.add_argument("--json", action="store_true")
    s = sub.add_parser("search", help="rechercher des passages sans générer de réponse")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=5)
    sub.add_parser("stats", help="statistiques du corpus et des chunks")
    args = ap.parse_args(argv)

    cfg = Config.depuis_yaml(args.config) if args.config else Config()
    pipe = Pipeline(cfg)
    if args.cmd == "stats":
        print(json.dumps({"documents": len(pipe.docs), "chunking": cfg.chunking, **stats(pipe.chunks)},
                         ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "search":
        for h in pipe.rechercher(args.question, args.k):
            print(f"{h.rang}. [{h.score:.3f}] {h.chunk.ref}{'  ⚠ injection suspecte' if h.chunk.injection_suspecte else ''}")
        return 0
    r = pipe.repondre(args.question)
    if args.json:
        print(json.dumps({"question": r.question, "reponse": r.reponse.texte, "abstention": r.reponse.abstention,
                          "confiance": round(r.confiance, 3), "citations": [h.chunk.ref for h in r.reponse.citations],
                          "ancrage": r.reponse.ancrage, "duree_ms": r.duree_ms}, ensure_ascii=False, indent=2))
        return 0
    print(r.reponse.texte)
    if r.reponse.citations:
        print("\nSources :")
        for i, h in enumerate(r.reponse.citations, 1):
            print(f"  [{i}] {h.chunk.doc_titre} › {h.chunk.section}")
    if args.explain:
        print(f"\nConfiance : {r.confiance:.2f} (seuil {cfg.seuil_abstention}) · {r.duree_ms} ms · "
              f"générateur {r.reponse.fournisseur} · injections écartées : {r.injections_ecartees}")
        print(f"Ancrage : {r.reponse.ancrage}")
        print("Passages candidats :")
        for h in r.candidats:
            print(f"  {h.rang}. [{h.score:.3f}] {h.chunk.ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
