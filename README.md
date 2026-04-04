# Predilex NLP — Classification de décisions judiciaires

Projet M2 de Deep NLP — Fine-tuning CamemBERT sur des décisions de cours d'appel françaises.

## Objectif

Pour chaque décision judiciaire, prédire automatiquement :
- **Sexe** de la victime (`homme` / `femme`)
- **Date de l'accident** (`YYYY-MM-DD` ou `n.c.`)
- **Date de consolidation** (`YYYY-MM-DD` ou `n.c.`)

## Dataset

770 décisions judiciaires de cours d'appel françaises (accidents du travail / droit social), issues du challenge **Predilex**.

| Fichier | Description |
|---|---|
| `Y_train_predilex.csv` | Labels : ID, sexe, date_accident, date_consolidation |
| `x_train_ids.csv` | Mapping ID → nom de fichier `.txt` |
| `txt_files/` | 770 décisions en texte brut (`{Ville}_{NumRG}.txt`) |

**Statistiques clés :**
- Sexe : 559 homme (73%) / 206 femme (27%) / 5 n.c.
- date_consolidation manquante dans 42% des cas
- Longueur moyenne : 3 315 mots par document (tous > 512 tokens)

## Architecture

```
Pipeline hybride NLP + Deep Learning
─────────────────────────────────────

Texte brut
  │
  ├─► [SEXE]  CamemBERT fine-tuné  →  homme / femme
  │           512 premiers tokens
  │           WeightedLoss (73/27)
  │
  └─► [DATES] Regex FR  →  phrases candidates
              CamemBERT fine-tuné  →  accident / consolidation / autre
              Seuil calibré  →  date ou n.c.
```

**Modèle de base :** `camembert-base` (110M params, entraîné sur 138 GB de français)

**Pourquoi CamemBERT ?**
- Spécifique au français → meilleure tokenisation du vocabulaire juridique
- Taille adaptée à 770 documents (un modèle plus grand overfitterait)
- Référence établie pour le NLP français

## Structure du projet

```
NLP S2/
├── configs/
│   └── config.yaml              # Tous les hyperparamètres et chemins
├── data/
│   ├── raw/                     # Données brutes (non versionnées)
│   │   ├── Y_train_predilex.csv
│   │   ├── x_train_ids.csv
│   │   └── txt_files/
│   └── processed/               # Sorties intermédiaires
├── models/                      # Modèles fine-tunés (non versionnés)
│   ├── sex_classifier/best_model/
│   └── date_classifier/best_model/
├── notebooks/
│   ├── EDA.ipynb                # Analyse exploratoire des données
│   └── Colab_Training.ipynb     # Entraînement sur Google Colab (GPU)
├── src/
│   ├── data_loader.py           # Chargement et split des données
│   ├── preprocessing.py         # Regex dates, nettoyage, datasets PyTorch
│   ├── model_utils.py           # Chargement/sauvegarde CamemBERT
│   ├── evaluate.py              # Métriques (phrase + document)
│   ├── train_sex.py             # Fine-tuning classification sexe
│   ├── train_dates.py           # Fine-tuning classification dates
│   └── predict.py               # Inférence finale + soumission
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/djema-rayane/predilex-nlp.git
cd predilex-nlp
pip install -r requirements.txt
```

Placer les données dans `data/raw/` (voir structure ci-dessus).

## Utilisation

### Analyse exploratoire
```bash
jupyter notebook notebooks/EDA.ipynb
```

### Entraînement (local, CPU lent — préférer Colab)
```bash
cd "NLP S2"

# Modèle sexe (~4h CPU / ~15 min GPU)
python src/train_sex.py

# Modèle dates (~25h CPU / ~45 min GPU)
python src/train_dates.py
```

### Entraînement sur Google Colab (recommandé)
Ouvrir `notebooks/Colab_Training.ipynb` sur [colab.research.google.com](https://colab.research.google.com) avec runtime **GPU T4**.
Durée totale : ~1h.

### Inférence
```bash
# Vérification sur train set
python src/predict.py --train-set

# Soumission finale (test set requis)
python src/predict.py
```

### Mode debug (test rapide du pipeline)
```bash
python src/train_sex.py --debug    # 50 docs, 2 epochs
python src/train_dates.py --debug  # 30 docs, 2 epochs
```

## Choix techniques

### Tâche 1 — Sexe

| Choix | Justification |
|---|---|
| `camembert-base` | French-specific, adapté à 770 docs |
| First 512 tokens | Signal (Monsieur/Madame) dans les 3.7% premiers du texte (médiane) |
| `WeightedTrainer` | Compense déséquilibre 73/27 via CrossEntropyLoss pondérée |
| Early stopping sur `f1_weighted` | Évite l'overfitting sur le petit dataset |

### Tâche 2+3 — Dates

| Choix | Justification |
|---|---|
| Regex + CamemBERT | Regex pour extraction déterministe, BERT pour classification sémantique |
| Contexte ±2 phrases | La sémantique de la date dépend du contexte environnant |
| 256 tokens max | Phrases courtes → batch size × 2, entraînement 2× plus rapide |
| Oversampling (ratio 0.3) | Corrige le ratio 1:25 accident/autre |
| Seuil calibré (0.4) | 42% de n.c. → le modèle doit savoir "ne pas répondre" |
| `f1_macro` comme métrique | Pénalise les erreurs sur les classes rares (accident, consolidation) |

## Résultats attendus

| Tâche | Métrique | Attendu |
|---|---|---|
| Sexe | Accuracy | >97% |
| Sexe | F1 Weighted | >96% |
| date_accident | Accuracy document | >80% |
| date_consolidation | Accuracy document | >70% |

## Dépendances principales

- `transformers>=4.38` — CamemBERT + Trainer HuggingFace
- `torch>=2.0` — Backend PyTorch
- `accelerate>=1.1` — Optimisations entraînement
- `scikit-learn>=1.3` — Métriques, split stratifié
- `sentencepiece>=0.1.99` — Tokeniser SentencePiece de CamemBERT
