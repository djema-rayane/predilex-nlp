# Predilex NLP — Extraction automatique d'informations dans des décisions judiciaires françaises

**Projet M2 Deep NLP** | Fine-tuning CamemBERT sur corpus juridique français

---

## 1. Contexte et problématique

### 1.1 Le challenge Predilex

Predilex est un projet de recherche en droit social visant à analyser automatiquement la jurisprudence française des cours d'appel. Les décisions de justice sont publiques mais non structurées : des milliers de pages de texte brut sans base de données exploitable. Un juriste met en moyenne 20 à 30 minutes pour lire et extraire les informations clés d'une seule décision, ce qui rend impossible toute analyse à grande échelle.

### 1.2 Objectif

L'objectif est d'automatiser l'extraction de trois informations dans chaque décision judiciaire relative à un accident du travail :

- **Sexe de la victime** : `homme` ou `femme`
- **Date de l'accident** : format `YYYY-MM-DD` ou `n.c.` si non trouvée
- **Date de consolidation** : format `YYYY-MM-DD` ou `n.c.` si non applicable

La date de consolidation correspond au moment où les blessures sont médicalement stabilisées. Elle est juridiquement et financièrement cruciale : avant la consolidation, la victime perçoit des indemnités journalières ; après, une rente viagère si des séquelles permanentes subsistent (IPP). Plus la consolidation est tardive, plus l'indemnisation est élevée.

---

## 2. Données et analyse exploratoire

### 2.1 Description du corpus

Le dataset comprend **770 décisions de cours d'appel françaises** (accidents du travail), issues du challenge Predilex. Les documents ont une longueur moyenne de **3 315 mots**, avec un maximum dépassant 10 000 mots. Fait crucial : **tous les documents dépassent 512 tokens**, ce qui constitue la limite dure des modèles Transformer et représente le principal défi technique.

| Caractéristique | Valeur |
|---|---|
| Nombre de documents | 770 |
| Longueur moyenne | 3 315 mots |
| Tous > 512 tokens | OUI |
| Juridictions | Agen, Paris, Versailles, Toulouse... |
| Période couverte | 1980 – 2020 |

### 2.2 Distribution des labels

**Sexe :** 559 hommes (73%), 206 femmes (27%), 5 non renseignés (exclus de l'évaluation). Le déséquilibre 73/27 est une contrainte technique importante : sans traitement, un modèle naïf apprend à prédire "homme" par défaut.

**Date d'accident :** présente dans 97% des documents (c'est le fait central du litige). Seulement 3% de `n.c.`

**Date de consolidation :** absente dans **42% des cas** (accident récent, ou juge non amené à se prononcer). C'est la tâche la plus difficile.

### 2.3 Observations clés de l'EDA

Quatre observations ont directement guidé les choix architecturaux :

1. **Le signal sexe apparaît très tôt.** La mention "Monsieur / Madame" se trouve en médiane à **3,7% du texte**. Les 512 premiers tokens capturent ce signal dans **94% des cas** → il suffit de lire le début du document pour prédire le sexe.

2. **Il y a en moyenne 33 dates par document.** Dates de jugement, de rapports médicaux, d'expertises... On ne peut pas donner les 3 300 mots au modèle. Il faut réduire le problème : extraire uniquement les phrases contenant des dates.

3. **Les formats de dates sont très variés.** Textuel ("9 avril 1991", "1er mars 2004"), numérique slash ("09/04/1991"), numérique tiret ("14-06-1999"), numérique point ("02.05.2000"), format liste ("- consolidation : 02.05.2000"), ou tronqué ("date de consolidation le" avec la date sur la ligne suivante).

4. **La consolidation suit des patterns récurrents** : mention directe avec mot-clé "consolidation" (70% des cas), ou formulation implicite via date d'examen médical final (30% des cas, très difficile à capturer).

---

## 3. Architecture et choix techniques

### 3.1 Choix du modèle de base : CamemBERT

**CamemBERT** est un modèle de langue pré-entraîné exclusivement sur du français (architecture RoBERTa, INRIA + Facebook AI Research). Il a été entraîné sur 138 GB de texte français avec 110 millions de paramètres et une tokenisation SentencePiece adaptée au français.

Ce choix est justifié face aux alternatives :

- **vs XLM-RoBERTa (multilingue)** : CamemBERT consacre 100% de son vocabulaire au français, ce qui donne une meilleure tokenisation des termes juridiques ("inexcusable", "consolidation", "indemnisation").
- **vs CamemBERT-large (330M params)** : avec 770 documents seulement, un modèle 3× plus grand entraînerait un overfitting quasi certain. La vraie différence vient du pipeline, pas de la taille.
- **vs LLMs (GPT, Mistral)** : nécessitent des GPU très puissants (80 GB VRAM) ou des API payantes. CamemBERT fine-tuné tourne sur un GPU T4 gratuit (Colab/Kaggle) et bat souvent un LLM en zero-shot sur des données spécialisées.

Le **Transfer Learning** permet de faire du Deep Learning efficace sur un petit corpus : CamemBERT a déjà appris la syntaxe et la sémantique du français sur 138 GB — les 770 documents servent uniquement à lui apprendre les patterns spécifiques des accidents du travail.

### 3.2 Pipeline hybride global

La contrainte des 512 tokens impose un pipeline hybride NLP classique + Deep Learning :

```
Document entier (3 300 mots)
         |
         |------------------------------------------|
         |                                          |
         v                                          v
[MODÈLE 1 — SEXE]                     [MODÈLE 2 — DATES]
512 premiers tokens                   Étape 1 : Regex
CamemBERT fine-tuné                   → ~33 phrases candidates
2 classes : homme / femme             → Normalisation YYYY-MM-DD
WeightedLoss (73/27)                          |
                                              v
                                    Étape 2 : CamemBERT fine-tuné
                                    Input : phrase + contexte ±2
                                    3 classes :
                                      - date_accident
                                      - date_consolidation
                                      - autre_date
                                              |
                                              v
                                    Étape 3 : Logique métier
                                    Score max par catégorie
                                    + Seuil de confiance
                                    → date ou n.c.
```

### 3.3 Modèle 1 — Classification du sexe

**Input :** 512 premiers tokens (stratégie *head truncation*)
**Output :** `homme` (0) ou `femme` (1)

Le nettoyage du texte préserve la casse (CamemBERT est sensible à "Monsieur" vs "monsieur") et retire les artefacts PDF (espaces insécables `\xa0`, form feeds `\x0c`, encodages Windows).

Pour gérer le déséquilibre 73/27, on utilise un **`WeightedTrainer`** qui surcharge la méthode `compute_loss` du Trainer HuggingFace pour appliquer une `CrossEntropyLoss` pondérée :

- Poids homme = `770 / (2 × 559)` ≈ **0,69** (classe fréquente, moins pénalisée)
- Poids femme = `770 / (2 × 206)` ≈ **1,87** (classe rare, 2,7× plus pénalisée)

Une erreur sur une femme compte donc 2,7× plus qu'une erreur sur un homme, forçant le modèle à apprendre les deux classes. L'**early stopping** sur le F1 Weighted (patience = 2 epochs) évite l'overfitting sur le petit dataset.

### 3.4 Modèle 2 — Extraction des dates (pipeline 3 étapes)

**Étape 1 — Extraction par Regex**

Une expression régulière couvre tous les formats de dates françaises rencontrés (textuel, slash, tiret). Pour chaque date extraite, on construit un contexte enrichi de ±2 phrases — crucial car la sémantique dépend souvent de la phrase précédente :

> *"L'expert a conclu à une consolidation."* ← phrase i-1
> *"La date retenue est le 7 novembre 1998."* ← phrase i (contient la date)

Sans la phrase i-1, le modèle ne sait pas que cette date est une consolidation. Toutes les dates sont normalisées vers le format ISO `YYYY-MM-DD`.

**Étape 2 — Classification CamemBERT**

Les phrases candidates (~25 000 au total pour 770 documents) sont classifiées en 3 classes. On utilise **256 tokens maximum** (au lieu de 512) : les phrases sont courtes, 256 tokens suffisent, et cela permet des batchs 2× plus grands.

Le déséquilibre est extrême : 3% `date_accident`, 2% `date_consolidation`, **95% `autre_date`**. Sans traitement, le modèle prédirait tout "autre_date" avec 95% d'accuracy mais 0% d'utilité. La solution est un **oversampling** des classes rares : on duplique les exemples minoritaires jusqu'à atteindre 30% de la classe majoritaire (`target_ratio=0.3`). On utilise également une `CrossEntropyLoss` pondérée et le **F1 Macro** comme métrique d'optimisation (qui pénalise également les erreurs sur toutes les classes, même les rares).

**Étape 3 — Logique métier et seuils**

Pour chaque document, on retient la phrase avec le meilleur score par catégorie, puis on applique un seuil de confiance :

- `threshold_accident = 0.40` : seuil bas car 97% des docs ont une vraie date d'accident
- `threshold_consolidation = 0.55` : seuil plus élevé car 42% des docs sont `n.c.` — le modèle doit être confiant pour ne pas inventer une date inexistante

---

## 4. Résultats et analyse

### 4.1 Résultats finaux

| Tâche | Baseline | Notre modèle | Gain |
|---|---|---|---|
| Sexe | 73,33% (toujours homme) | **90,67%** | **+17 pts** |
| date_accident | 30,67% (regex + 1ère occurrence) | **77,92%** | **+47 pts** |
| date_consolidation | 69,33% (regex + mot-clé) | **74,03%** | **+5 pts** |

### 4.2 Analyse des gains

**Sexe (+17 pts) :** le modèle a appris les signaux textuels (Monsieur/Madame, prénoms, pronoms). La WeightedLoss lui a permis d'atteindre **95% de recall sur les femmes** (vs 0% pour la baseline "toujours homme").

**Accident (+47 pts) :** c'est le gain le plus significatif. Le mot "accident" apparaît des dizaines de fois par document. La baseline prend la première occurrence → souvent une mauvaise date. CamemBERT comprend quelle phrase *définit* la date d'accident vs celles qui y font simplement *référence*.

**Consolidation (+5 pts) :** gain plus modeste car la baseline mots-clés est déjà forte (le mot "consolidation" est très spécifique). CamemBERT améliore surtout les cas implicites et la prédiction `n.c.` (97% vs 88% pour la baseline).

### 4.3 Analyse des erreurs

**Sur l'accident :**
- Confusion avec les dates de jugement portant sur le même accident
- Dates formulées sans jour exact ("victime d'un accident survenu en 1991") → le regex extrait une date incomplète
- La matrice de confusion niveau phrase montre 143 phrases "autre_date" classées comme "accident" (faux positifs), partiellement compensés par la logique d'agrégation

**Sur la consolidation :**
- Formulations implicites : date d'examen médical final sans le mot "consolidation"
- Phrases tronquées : "- date de consolidation le" avec la date sur la ligne suivante, non capturée par le contexte ±2
- Documents avec plusieurs dates de consolidation discutées (successives ou contestées)

### 4.4 Limites et pistes d'amélioration

| Limite | Solution envisagée |
|---|---|
| Précision phrase-level faible (53% accident) | Normale avec déséquilibre 1:25 — compenser par logique d'agrégation |
| Phrases de consolidation tronquées | Fusionner les lignes incomplètes avant extraction |
| Formulations implicites de consolidation | Fenêtre de contexte asymétrique (before=3, after=1) |
| Petit corpus (770 docs) | Levier principal : plus de données (5 000+ docs amélioreraient significativement) |

---

## 5. Conclusion

Ce projet a produit un pipeline complet d'extraction d'informations dans des documents juridiques longs, en répondant à deux défis majeurs : la limite des 512 tokens (résolue par le pipeline hybride Regex + CamemBERT) et le déséquilibre des classes (résolu par WeightedLoss et oversampling).

Le Transfer Learning sur CamemBERT permet de faire du Deep Learning efficace sur seulement 770 documents, en s'appuyant sur la compréhension du français acquise sur 138 GB de texte. Les résultats (+47 points sur la date d'accident) valident l'approche hybride : NLP classique pour la réduction du problème, Deep Learning pour la compréhension sémantique.

Les poids des modèles sont disponibles sur `models/sex_classifier/best_model/` et `models/date_classifier/best_model/`, reproductibles via `notebooks/Colab_Training.ipynb` (GPU T4, ~1h d'entraînement).
