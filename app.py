"""Interface de démonstration : question, réponse citée, passages, confiance et ancrage.

Lancer : PYTHONPATH=src streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ragdoc.pipeline import Config, Pipeline  # noqa: E402

st.set_page_config(page_title="Assistant documentaire RAG", page_icon="📚", layout="wide")


@st.cache_resource
def charger(chemin: str) -> Pipeline:
    return Pipeline(Config.depuis_yaml(chemin))


st.title("📚 Assistant documentaire interne")
st.caption("Entreprise de transport fictive · 25 procédures · réponses citées, abstention quand la documentation ne répond pas")

with st.sidebar:
    cfg_path = st.selectbox("Configuration", ["config/default.yaml", "config/dense_e5.yaml"])
    pipe = charger(cfg_path)
    st.write(f"**Découpage :** {pipe.config.chunking} · **{len(pipe.chunks)} chunks**")
    st.write(f"**Recherche :** {pipe.principal.nom}")
    st.write(f"**Générateur :** {pipe.generateur.nom}")
    seuil = st.slider("Seuil d'abstention", 0.0, 1.0, float(pipe.config.seuil_abstention), 0.01)
    pipe.config.seuil_abstention = seuil
    st.markdown("Exemples :")
    exemples = ["Quelle est la durée maximale de conduite journalière ?", "montant de lindemnité de repa",
                "Comment joindre l'astreinte atelier en cas de panne ?", "Quel est le salaire de base d'un conducteur ?"]
    choix = st.radio("", exemples, label_visibility="collapsed")

question = st.text_input("Votre question", value=choix)
if question:
    r = pipe.repondre(question)
    c1, c2 = st.columns([3, 2])
    with c1:
        if r.reponse.abstention:
            st.warning(r.reponse.texte)
        else:
            st.success(r.reponse.texte)
            for i, h in enumerate(r.reponse.citations, 1):
                with st.expander(f"[{i}] {h.chunk.doc_titre} › {h.chunk.section}"):
                    st.write(h.chunk.text.split("\n", 1)[-1])
    with c2:
        st.metric("Confiance", f"{r.confiance:.2f}", f"seuil {seuil}", delta_color="off")
        st.metric("Latence", f"{r.duree_ms} ms")
        anc = r.reponse.ancrage
        st.metric("Phrases ancrées", f"{anc['phrases_ancrees']:.0%}")
        if anc["nombres_non_ancres"]:
            st.error(f"Chiffres non trouvés dans les sources : {', '.join(anc['nombres_non_ancres'])}")
        if r.injections_ecartees:
            st.info(f"{r.injections_ecartees} passage(s) suspect(s) d'injection écarté(s)")
        st.markdown("**Passages candidats**")
        for h in r.candidats:
            st.write(f"{h.rang}. `{h.chunk.ref}` — score {h.score:.3f}")
