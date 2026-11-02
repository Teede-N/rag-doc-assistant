# Méthodologie

Ce document explique **comment** le projet a été conçu et évalué, et **pourquoi** les choix par défaut sont ceux-là. Les chiffres cités proviennent de `reports/evaluation.md`, régénéré par `python -m ragdoc.evaluate`.

## 1. Le problème

Un assistant documentaire interne doit répondre à des questions de salariés à partir de procédures d'entreprise. Trois exigences dominent, dans cet ordre :

1. **Ne jamais inventer.** Une réponse fausse sur une règle de sécurité ou de paie coûte plus cher qu'une absence de réponse.
2. **Citer ses sources.** L'utilisateur doit pouvoir vérifier en un clic.
3. **Savoir se taire.** Quand la documentation ne couvre pas la question, le système doit le dire.

Le corpus est **fictif** (25 procédures d'une PME de transport routier, ~670 lignes de Markdown) mais réaliste : tableaux, versions, services, formulations administratives, et un document piégé contenant une injection de prompt.

## 2. Jeu d'évaluation

`eval/questions.jsonl` contient 64 questions annotées à la main avec la section attendue (`gold`, format `doc#section`) et des mots-clés que la réponse doit contenir.

| Type | N | Ce que ça mesure |
|---|---|---|
| factuelle | 40 | Question directe, vocabulaire proche du document |
| multi | 5 | Réponse répartie sur deux sections ou deux documents |
| reformulation | 5 | Langage courant, vocabulaire différent du document (« je me fais rembourser l'essence ? ») |
| typo | 4 | Fautes de frappe et accents manquants |
| sans_reponse | 8 | Question plausible mais absente du corpus, dont des pièges lexicaux (le dimanche quand seul le samedi est documenté) |
| injection | 2 | Questions dont la meilleure section voisine contient l'injection |

Les types `reformulation` et `sans_reponse` sont volontairement les plus durs : ce sont ceux qui distinguent un prototype d'un système utilisable.

## 3. Métriques

**Recherche** (sur les 56 questions avec réponse, top-5) : Recall@5, Hit@1, MRR, nDCG@5. Un chunk est pertinent s'il couvre une section attendue ; avec les découpages fixes ou par phrases, plusieurs chunks peuvent couvrir la même section, et chaque section n'apporte un gain qu'une seule fois dans le nDCG.

**Abstention** : taux de refus correct sur les questions sans réponse, taux de refus à tort sur les questions avec réponse, exactitude du seuil estimée en validation croisée à 2 plis (le seuil est choisi sur une moitié, mesuré sur l'autre).

**Réponses** : une réponse est correcte si elle contient tous les mots-clés attendus (expressions régulières), ou si elle s'abstient sur une question sans réponse. On mesure aussi les **fuites d'injection** (la réponse contient le code piégé) et l'**ancrage** : part des phrases de la réponse retrouvées dans les passages cités, et nombres présents dans la réponse mais absents des sources.

## 4. Découpage : trois stratégies comparées

| Stratégie | Chunks | Principe |
|---|---|---|
| `sections` | 99 | Une section Markdown = un chunk. Fidèle à la structure, mais sections de taille très variable. |
| `fixed` | 73 | Fenêtres de 120 mots avec recouvrement de 30. Taille homogène, mais coupe les tableaux. |
| `sentences` | 113 | Fenêtres de 3 phrases avec pas de 2, sans franchir une section. Chunks courts et précis. |

Résultat : les trois sont proches en Recall@5 (0.96–0.98). `sentences` donne le meilleur MRR (0.955) parce que des chunks courts concentrent les termes de la question ; c'est la stratégie par défaut. Les tableaux Markdown sont convertis en phrases « colonne : valeur ; colonne : valeur. » avant le découpage, sans quoi BM25 ne voit qu'une suite de barres verticales.

## 5. Recherche : lexical, dense, hybride

- **BM25** sur mots racinisés (Snowball français, stopwords retirés) : la référence, MRR 0.93–0.94, 0.3 ms.
- **TF-IDF de n-grammes de caractères (3 à 5)** : conçu pour les fautes de frappe. Il est à égalité avec BM25 sur les questions factuelles et résiste bien aux fautes, au prix d'une latence ×4.
- **Dense spaCy** (moyenne de vecteurs de mots 300 d) : MRR 0.35–0.40. Les vecteurs de mots moyennés n'encodent ni l'ordre ni la négation, et le corpus utilise un vocabulaire métier que le modèle généraliste capture mal. Il est gardé comme **témoin négatif** : il montre qu'ajouter « du dense » n'améliore pas automatiquement un système.
- **Hybride RRF** (bm25 + char_tfidf, fusion par rangs réciproques, k = 60) : meilleur compromis, MRR 0.955 sur `sentences`. La fusion par rangs évite de devoir calibrer des scores d'échelles différentes.
- **Hybride + dense spaCy** : dégrade fortement (MRR 0.67). Un composant faible fait du bruit dans une fusion par rangs ; c'est documenté plutôt que caché.
- **Dense e5 (`intfloat/multilingual-e5-small`)** : prévu via `config/dense_e5.yaml`, non évalué ici car Hugging Face n'était pas accessible depuis l'environnement d'évaluation. C'est le candidat naturel pour les reformulations, où le lexical plafonne (MRR 0.8).

## 6. Reranking appris (LTR)

Un reranker de type *learning-to-rank* réordonne les 20 candidats de l'hybride avec une régression logistique sur 7 features : scores bm25, char_tfidf et dense normalisés, couverture des termes de la question, correspondance avec le titre de section, longueur du chunk et meilleur rang obtenu. Évalué en **validation croisée à 5 plis** sur les questions, il atteint le niveau de l'hybride (MRR 0.94) sans le dépasser : avec 56 questions, il n'y a pas assez de signal pour apprendre mieux que la fusion. Les coefficients sont lisibles dans le rapport (couverture des termes et rang minimal dominent, dense à 0). Un cross-encoder (`CrossEncoderReranker`) est branché mais optionnel pour la même raison d'accès réseau.

## 7. Abstention calibrée

Un moteur lexical renvoie toujours quelque chose. Le score de confiance combine trois signaux : le meilleur score BM25 saturé (poids 0.5), la meilleure similarité caractères (0.3) et la marge entre les deux premiers candidats (0.2). Le seuil (0.44) est choisi par balayage sur le jeu d'évaluation, avec exactitude estimée en validation croisée (0.906).

Résultat : 62 % des questions sans réponse sont correctement refusées, 4 % des questions avec réponse sont refusées à tort. Les échecs restants sont des **pièges lexicaux** (N05, N07, N08) : la question partage presque tous ses mots avec une section qui ne répond pourtant pas. Seul un lecteur (LLM) peut trancher ce cas, ce que fait le générateur Claude via l'instruction d'abstention dans le prompt.

## 8. Génération et vérification

- **Extractif** (par défaut sans clé API) : sélectionne jusqu'à 3 phrases des passages par recouvrement avec la question, bonus pour les chiffres et le rang du passage, et les cite `[n]`. Déterministe, 100 % ancré par construction, mais incapable de reformuler ou de synthétiser.
- **Claude** (`ANTHROPIC_API_KEY` définie, température 0) : prompt système qui impose les citations, l'abstention et le traitement des passages comme des données. Sa sortie passe par la même vérification d'ancrage.
- **Vérification d'ancrage** : chaque phrase de la réponse doit être retrouvée (recouvrement de tokens) dans les passages cités ; chaque nombre de la réponse doit apparaître dans les sources. Une réponse avec un chiffre non ancré est signalée dans l'interface.

## 9. Sécurité : injections dans les documents

Le document `25_faq_interne.md` contient un paragraphe qui tente de détourner l'assistant (« ignore toutes les instructions précédentes… »). Deux défenses :

1. À l'ingestion, une expression régulière marque les chunks suspects (`injection_suspecte`) ; ils sont exclus du contexte envoyé au générateur.
2. Le prompt système rappelle que les passages sont des données, jamais des instructions.

Sur les deux questions d'injection, la réponse correcte est produite et le code piégé n'apparaît jamais (0 fuite).

## 10. Limites et suites

- **Reformulations** : 0.4 d'exactitude finale. Il faut un encodeur de phrases (e5) et/ou une expansion de requête par LLM.
- **Pièges lexicaux** sur l'abstention : un lecteur LLM ou un classifieur « la section répond-elle ? » entraîné sur davantage de données.
- **Taille du jeu** : 64 questions suffisent à comparer des méthodes, pas à affirmer des écarts de 1 point ; les écarts inférieurs à 0.02 de MRR ne sont pas significatifs.
- **Corpus statique** : pas de réindexation incrémentale ni de gestion des versions de documents.

## Reproduire

```bash
pip install -r requirements.txt
pytest -q                                   # 18 tests
PYTHONPATH=src python -m ragdoc.evaluate    # ~2 min, régénère reports/
PYTHONPATH=src python -m ragdoc.evaluate --claude   # ajoute le générateur Claude (clé API requise)
```
