"""
data_loader.py
--------------
Chargement, validation et split des données Predilex.

Responsabilités :
  - Lire les fichiers CSV (y_train, x_train_ids)
  - Charger le texte brut de chaque décision (.txt) avec gestion d'encodage
  - Retourner un DataFrame unifié : filename | text | sexe | date_accident | date_consolidation
  - Fournir les splits train / validation / test (stratifiés, sans data leakage)
  - Calculer les poids de classe pour les tâches déséquilibrées

Structure des données Predilex :
  data/raw/
    Y_train_predilex.csv      → ID, sexe, date_accident, date_consolidation
    x_train_ids.csv           → ID, filename
    txt_files/
      Agen_100515.txt         → texte brut de la décision juridique
      ...                       (770 fichiers, nommés {Ville}_{NumRG}.txt)

Particularités de la donnée :
  - 770 documents (petit dataset → attention à l'overfitting)
  - sexe : 559 homme / 206 femme / 5 n.c. (les n.c. ne peuvent pas être stratifiés)
  - date_consolidation : 42% de n.c. / n.a. (classe très déséquilibrée)
  - Encodage des fichiers .txt : utf-8 ou latin-1 / cp1252 (selon juridiction)
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """
    Charge le fichier de configuration YAML.

    Le fichier config.yaml centralise tous les hyperparamètres et chemins.
    On l'utilise partout plutôt que des valeurs hardcodées dans le code.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Fichier de config introuvable : {config_path}\n"
            "Assurez-vous de lancer le script depuis la racine du projet (NLP S2/)."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Configuration chargée depuis : {config_path}")
    return cfg


# =============================================================================
# Lecture des fichiers texte
# =============================================================================

def read_text_file(filepath: str) -> str:
    """
    Lit un fichier texte en gérant les encodages courants.

    Les décisions judiciaires proviennent de différentes juridictions et
    peuvent être encodées différemment (utf-8 pour les plus récentes,
    latin-1 / cp1252 pour les plus anciennes ou celles de DOM-TOM).

    On essaie les encodages dans l'ordre, du plus strict au plus permissif.
    """
    encodings = ("utf-8", "latin-1", "cp1252")
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    # Dernier recours : ignore les caractères illisibles
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        logger.warning(f"Encodage forcé (utf-8 + replace) pour : {filepath}")
        return f.read()


# =============================================================================
# Chargement des données d'entraînement
# =============================================================================

def load_train_data(config: dict) -> pd.DataFrame:
    """
    Charge et assemble les données d'entraînement.

    Étapes :
      1. Charge Y_train_predilex.csv (labels)
      2. Charge x_train_ids.csv (mapping ID → fichier)
      3. Joint les deux sur l'ID
      4. Charge le texte brut de chaque décision
      5. Filtre les documents vides (si fichier manquant)

    Returns
    -------
    DataFrame avec colonnes :
        filename | text | sexe | date_accident | date_consolidation
    """
    y_path = Path(config["paths"]["y_train_file"])
    ids_path = Path(config["paths"]["x_train_ids_file"])
    txt_dir = Path(config["paths"]["x_train_txt_dir"])

    # Vérifications
    for p in (y_path, ids_path, txt_dir):
        if not p.exists():
            raise FileNotFoundError(
                f"Chemin introuvable : {p}\n"
                "Vérifiez que les données sont dans data/raw/ "
                "ou modifiez les chemins dans configs/config.yaml"
            )

    # --- Chargement des labels ---
    y_train = pd.read_csv(y_path, index_col="ID")
    logger.info(f"Y_train chargé : {len(y_train)} lignes | "
                f"colonnes : {list(y_train.columns)}")

    # --- Chargement du mapping ID → filename ---
    x_ids = pd.read_csv(ids_path, index_col="ID")
    logger.info(f"x_train_ids chargé : {len(x_ids)} lignes")

    # --- Jointure sur l'index ID ---
    df = x_ids.join(y_train, how="inner")
    if len(df) != len(x_ids):
        logger.warning(
            f"Jointure imparfaite : {len(x_ids)} IDs dans x_ids, "
            f"{len(df)} après jointure avec y_train"
        )
    logger.info(f"Jointure OK : {len(df)} documents")

    # --- Chargement des textes ---
    texts = []
    missing = []
    for _, row in df.iterrows():
        filepath = txt_dir / row["filename"]
        if filepath.exists():
            texts.append(read_text_file(str(filepath)))
        else:
            texts.append("")
            missing.append(row["filename"])

    if missing:
        logger.warning(
            f"{len(missing)} fichiers introuvables dans {txt_dir} : "
            f"{missing[:3]}{'...' if len(missing) > 3 else ''}"
        )

    df = df.copy()
    df["text"] = texts

    # Filtrer les documents vides
    n_before = len(df)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    if len(df) < n_before:
        logger.warning(f"{n_before - len(df)} documents vides supprimés")

    logger.info(
        f"Données d'entraînement prêtes : {len(df)} documents | "
        f"sexe : {df['sexe'].value_counts().to_dict()} | "
        f"date_acc n.c. : {(df['date_accident'].isin(['n.c.','n.a.'])).sum()} | "
        f"date_cons n.c. : {(df['date_consolidation'].isin(['n.c.','n.a.'])).sum()}"
    )
    return df


# =============================================================================
# Chargement des données de test (inférence)
# =============================================================================

def load_test_data(config: dict) -> pd.DataFrame:
    """
    Charge les données de test (sans labels).

    Utilisé pour générer le fichier de soumission final.

    Returns
    -------
    DataFrame avec colonnes : filename | text
    """
    ids_path = Path(config["paths"]["x_test_ids_file"])
    txt_dir = Path(config["paths"]["x_test_txt_dir"])

    for p in (ids_path, txt_dir):
        if not p.exists():
            raise FileNotFoundError(
                f"Données de test introuvables : {p}\n"
                "Vérifiez que les fichiers de test sont disponibles."
            )

    x_ids = pd.read_csv(ids_path, index_col="ID")
    logger.info(f"x_test_ids chargé : {len(x_ids)} lignes")

    texts = []
    missing = []
    for _, row in x_ids.iterrows():
        filepath = txt_dir / row["filename"]
        if filepath.exists():
            texts.append(read_text_file(str(filepath)))
        else:
            texts.append("")
            missing.append(row["filename"])

    if missing:
        logger.warning(f"{len(missing)} fichiers de test introuvables : {missing[:3]}")

    x_ids = x_ids.copy()
    x_ids["text"] = texts
    x_ids = x_ids[x_ids["text"].str.strip() != ""].reset_index(drop=True)
    logger.info(f"Données de test prêtes : {len(x_ids)} documents")
    return x_ids


# =============================================================================
# Split train / validation / test
# =============================================================================

def split_train_val_test(
    df: pd.DataFrame,
    val_split: float = 0.15,
    test_split: float = 0.10,
    seed: int = 42,
    label_col: str = "sexe",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Découpe stratifiée train / validation / test au niveau DOCUMENT.

    Pourquoi stratifier ?
      → Assurer que la distribution des classes est identique dans chaque split.
      → Crucial avec un petit dataset (770 docs) : un split non-stratifié
        pourrait déséquilibrer les classes par hasard.

    Gestion des labels n.c. :
      → Les 5 documents avec sexe='n.c.' ne peuvent pas être stratifiés
        (sklearn lève une erreur si une classe a < 2 membres).
      → Ces documents sont forcés dans train uniquement.

    Pourquoi splitter au niveau DOCUMENT (et non phrase) ?
      → Éviter le data leakage : si les phrases du même document sont
        réparties entre train et val, le modèle "voit" le même document
        pendant entraînement et évaluation → métriques artificellement gonflées.

    Parameters
    ----------
    df        : DataFrame complet
    val_split : proportion de validation (sur le total)
    test_split: proportion de test interne (sur le total)
    seed      : reproductibilité
    label_col : colonne pour la stratification

    Returns
    -------
    train_df, val_df, test_df
    """
    # Séparer les exemples non-stratifiables (n.c., n.a., vide)
    nc_values = {"n.c.", "n.a.", ""}
    if label_col in df.columns:
        nc_mask = df[label_col].isin(nc_values) | df[label_col].isna()
        df_nc = df[nc_mask].copy()
        df_valid = df[~nc_mask].copy()
        if len(df_nc) > 0:
            logger.info(
                f"{len(df_nc)} exemples avec {label_col}=n.c./NaN → "
                "ajoutés au train uniquement (non-stratifiables)"
            )
    else:
        df_nc = pd.DataFrame()
        df_valid = df.copy()

    stratify = df_valid[label_col] if label_col in df_valid.columns else None

    # Split 1 : extraire le test set
    train_val, test_df = train_test_split(
        df_valid,
        test_size=test_split,
        random_state=seed,
        stratify=stratify,
    )

    # Split 2 : extraire le validation set du reste
    # Ajustement du ratio : val_split / (1 - test_split)
    val_ratio_adjusted = val_split / (1.0 - test_split)
    stratify_tv = train_val[label_col] if label_col in train_val.columns else None

    train_df, val_df = train_test_split(
        train_val,
        test_size=val_ratio_adjusted,
        random_state=seed,
        stratify=stratify_tv,
    )

    # Ajouter les n.c. au train
    if len(df_nc) > 0:
        train_df = pd.concat([train_df, df_nc], ignore_index=True).sample(
            frac=1, random_state=seed
        )

    logger.info(
        f"Split final — Train: {len(train_df)} | "
        f"Val: {len(val_df)} | Test: {len(test_df)}"
    )

    # Vérification de la distribution des classes
    if label_col in df.columns:
        for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            dist = split[label_col].value_counts(normalize=True).round(3).to_dict()
            logger.info(f"  {name} {label_col} distribution : {dist}")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# =============================================================================
# Poids de classe
# =============================================================================

def get_class_weights(
    df: pd.DataFrame,
    label_col: str,
    label2id: dict,
) -> list:
    """
    Calcule les poids de classe inversement proportionnels à leur fréquence.

    Formule : weight_i = total / (n_classes × count_i)

    Utilisé dans train_sex.py pour compenser le déséquilibre 73/27 (homme/femme).
    Ces poids sont passés à la CrossEntropyLoss pour pénaliser davantage
    les erreurs sur la classe minoritaire (femme).

    Returns
    -------
    Liste de floats dans l'ordre des IDs (index = label_id)
    """
    n_classes = len(label2id)
    total = len(df)
    valid_df = df[df[label_col].isin(label2id.keys())]

    weights = []
    for label_name, label_id in sorted(label2id.items(), key=lambda x: x[1]):
        count = (valid_df[label_col] == label_name).sum()
        if count == 0:
            logger.warning(f"Classe '{label_name}' absente → poids=1.0")
            weights.append(1.0)
        else:
            w = total / (n_classes * count)
            weights.append(round(w, 4))

    logger.info(f"Poids de classe ({label_col}) : "
                f"{dict(zip(label2id.keys(), weights))}")
    return weights


# =============================================================================
# Test rapide
# =============================================================================

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    cfg = load_config(config_path)

    print("\n=== Chargement des données d'entraînement ===")
    df = load_train_data(cfg)

    print("\n--- Aperçu ---")
    print(df[["filename", "sexe", "date_accident", "date_consolidation"]].head(10))

    print(f"\n--- Statistiques ---")
    print(f"Nombre de documents : {len(df)}")
    print(f"Longueur texte (moy) : {df['text'].str.len().mean():,.0f} chars")
    print(f"\nDistribution sexe :\n{df['sexe'].value_counts()}")
    print(f"\ndate_accident n.c. : {df['date_accident'].isin(['n.c.','n.a.']).sum()} / {len(df)}")
    print(f"date_consolidation n.c. : {df['date_consolidation'].isin(['n.c.','n.a.']).sum()} / {len(df)}")

    print("\n=== Split train/val/test ===")
    train_df, val_df, test_df = split_train_val_test(
        df,
        val_split=cfg["sex_classification"]["val_split"],
        test_split=cfg["sex_classification"]["test_split"],
        seed=cfg["sex_classification"]["seed"],
        label_col="sexe",
    )

    print("\n=== Poids de classe (sexe) ===")
    weights = get_class_weights(train_df, "sexe", cfg["sex_classification"]["label2id"])
    print(f"Poids : {weights}")
