"""
predict.py
----------
Inférence finale sur le test set et génération du fichier de soumission.

Ce script orchestre l'inférence complète :
  1. Charge les deux modèles fine-tunés (sexe + dates)
  2. Pour chaque document du test set :
       a. Prédit le sexe avec le modèle sexe
       b. Extrait les dates candidates (regex)
       c. Classe chaque phrase candidate (modèle dates)
       d. Applique la logique métier (argmax + seuil consolidation)
  3. Génère le fichier de soumission CSV

Format du fichier de soumission :
  ID,sexe,date_accident,date_consolidation
  0,homme,1991-04-09,n.c.
  1,femme,2005-06-10,2010-01-19
  ...

Prérequis :
  - python src/train_sex.py doit avoir été lancé
  - python src/train_dates.py doit avoir été lancé
  - Les modèles doivent être dans models/sex_classifier/best_model/
    et models/date_classifier/best_model/

Usage :
  cd "NLP S2"
  python src/predict.py                         # inférence sur le test set
  python src/predict.py --config configs/config.yaml
  python src/predict.py --train-set             # inférence sur train (vérification)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_config, load_test_data, load_train_data
from src.model_utils import (
    get_device,
    load_trained_model,
    load_tokenizer,
    predict_batch_with_probs,
)
from src.preprocessing import (
    SexDataset,
    clean_text,
    extract_date_sentences,
)
from src.train_dates import predict_dates_for_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Inférence sexe
# =============================================================================

def predict_sex_for_documents(
    df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    id2label: dict,
    max_length: int = 512,
    truncation_strategy: str = "head",
    batch_size: int = 8,
) -> pd.DataFrame:
    """
    Prédit le sexe de la victime pour chaque document.

    Traitement par batch pour éviter les OOM sur GPU.
    La stratégie de truncation (head/head_tail) doit être identique
    à celle utilisée pendant l'entraînement.

    Parameters
    ----------
    df                  : DataFrame avec colonne 'text'
    model               : CamemBERT fine-tuné pour le sexe
    tokenizer           : tokenizer CamemBERT
    device              : cuda ou cpu
    id2label            : {0: "homme", 1: "femme"}
    max_length          : 512 tokens
    truncation_strategy : "head" ou "head_tail"
    batch_size          : taille des batches d'inférence

    Returns
    -------
    df avec colonnes 'sexe_pred' et 'sexe_proba'
    """
    from src.preprocessing import truncate_head_tail

    logger.info(f"Inférence sexe sur {len(df)} documents...")
    texts = df["text"].tolist()

    # Appliquer la même stratégie de truncation qu'à l'entraînement
    if truncation_strategy == "head_tail":
        processed = [truncate_head_tail(clean_text(t), tokenizer, max_length) for t in texts]
    else:
        processed = [clean_text(t) for t in texts]

    preds, probs = predict_batch_with_probs(
        processed, model, tokenizer, device,
        max_length=max_length, batch_size=batch_size
    )

    df = df.copy()
    df["sexe_pred"]  = [id2label[p] for p in preds]
    df["sexe_proba"] = [float(probs[i, preds[i]]) for i in range(len(preds))]

    # Stats
    counts = pd.Series([id2label[p] for p in preds]).value_counts()
    logger.info(f"Distribution prédite : {counts.to_dict()}")
    low_conf = (df["sexe_proba"] < 0.7).sum()
    if low_conf > 0:
        logger.warning(
            f"{low_conf} prédictions de sexe avec confiance < 0.7 "
            "(vérifier manuellement si possible)"
        )
    return df


# =============================================================================
# Inférence dates
# =============================================================================

def predict_dates_for_documents(
    df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    label2id: dict,
    context_window: int = 2,
    max_length: int = 256,
    consolidation_threshold: float = 0.4,
) -> pd.DataFrame:
    """
    Prédit date_accident et date_consolidation pour chaque document.

    Utilise predict_dates_for_document() de train_dates.py
    (même logique que l'évaluation pendant l'entraînement).

    Parameters
    ----------
    df                       : DataFrame avec colonne 'text' et 'filename'
    model                    : CamemBERT fine-tuné pour les dates
    tokenizer                : tokenizer CamemBERT
    device                   : cuda ou cpu
    label2id                 : {"date_accident": 0, "date_consolidation": 1, ...}
    context_window           : fenêtre de contexte (identique à l'entraînement)
    max_length               : 256 tokens
    consolidation_threshold  : seuil de confiance pour n.c.

    Returns
    -------
    df avec colonnes 'date_accident_pred', 'date_consolidation_pred'
    """
    logger.info(
        f"Inférence dates sur {len(df)} documents "
        f"(seuil consolidation={consolidation_threshold})..."
    )

    date_acc_preds = []
    date_cons_preds = []
    acc_scores = []
    cons_scores = []

    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(df)} documents traités...")

        result = predict_dates_for_document(
            text=row["text"],
            model=model,
            tokenizer=tokenizer,
            device=device,
            label2id=label2id,
            context_window=context_window,
            max_length=max_length,
            consolidation_threshold=consolidation_threshold,
        )
        date_acc_preds.append(result["date_accident"])
        date_cons_preds.append(result["date_consolidation"])
        acc_scores.append(result.get("_acc_score", 0.0))
        cons_scores.append(result.get("_cons_score", 0.0))

    df = df.copy()
    df["date_accident_pred"]       = date_acc_preds
    df["date_consolidation_pred"]  = date_cons_preds
    df["_acc_score"]               = acc_scores
    df["_cons_score"]              = cons_scores

    # Stats
    nc_acc  = (pd.Series(date_acc_preds) == "n.c.").sum()
    nc_cons = (pd.Series(date_cons_preds) == "n.c.").sum()
    logger.info(
        f"Dates prédites : "
        f"accident n.c.={nc_acc} ({nc_acc/len(df)*100:.1f}%) | "
        f"consolidation n.c.={nc_cons} ({nc_cons/len(df)*100:.1f}%)"
    )
    return df


# =============================================================================
# Génération du fichier de soumission
# =============================================================================

def generate_submission(
    df: pd.DataFrame,
    output_path: str,
    id_col: str = "ID",
) -> pd.DataFrame:
    """
    Génère le fichier CSV de soumission au format Predilex.

    Format attendu :
        ID,sexe,date_accident,date_consolidation
        0,homme,1991-04-09,n.c.
        ...

    Parameters
    ----------
    df          : DataFrame avec colonnes sexe_pred, date_accident_pred,
                  date_consolidation_pred et l'ID
    output_path : chemin du fichier CSV de sortie
    id_col      : nom de la colonne ID dans df
    """
    submission = pd.DataFrame({
        "ID":                  df.index if id_col not in df.columns else df[id_col],
        "sexe":                df["sexe_pred"],
        "date_accident":       df["date_accident_pred"],
        "date_consolidation":  df["date_consolidation_pred"],
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    logger.info(f"Fichier de soumission sauvegardé : {output_path}")
    logger.info(f"  Lignes : {len(submission)}")
    logger.info(f"  Aperçu :\n{submission.head(5).to_string(index=False)}")

    return submission


# =============================================================================
# Main
# =============================================================================

def main():
    # --------------------------------------------------------------------------
    # Arguments CLI
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Inférence finale Predilex — génération de la soumission"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Chemin vers le fichier de configuration YAML"
    )
    parser.add_argument(
        "--train-set", action="store_true",
        help="Lancer l'inférence sur le train set (vérification)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Seuil de consolidation (override la valeur du config)"
    )
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------
    cfg = load_config(args.config)
    sex_cfg  = cfg["sex_classification"]
    date_cfg = cfg["date_classification"]

    device = get_device()
    threshold = args.threshold if args.threshold is not None else date_cfg["threshold_consolidation"]

    # --------------------------------------------------------------------------
    # Chargement des modèles
    # --------------------------------------------------------------------------
    logger.info("=== Chargement des modèles ===")

    sex_model_path  = os.path.join(sex_cfg["output_dir"], "best_model")
    date_model_path = os.path.join(date_cfg["output_dir"], "best_model")

    logger.info(f"Modèle sexe  : {sex_model_path}")
    logger.info(f"Modèle dates : {date_model_path}")

    tokenizer  = load_tokenizer(cfg["model"]["name"])
    sex_model  = load_trained_model(sex_model_path,  num_labels=sex_cfg["num_labels"])
    date_model = load_trained_model(date_model_path, num_labels=date_cfg["num_labels"])

    sex_model.to(device)
    date_model.to(device)

    # --------------------------------------------------------------------------
    # Chargement des données
    # --------------------------------------------------------------------------
    if args.train_set:
        logger.info("=== Inférence sur le train set (vérification) ===")
        df = load_train_data(cfg)
        output_path = os.path.join(
            cfg["paths"]["processed_data_dir"], "submission_train_check.csv"
        )
    else:
        logger.info("=== Inférence sur le test set ===")
        df = load_test_data(cfg)
        output_path = cfg["paths"]["submission_file"]

    logger.info(f"Documents à traiter : {len(df)}")

    # --------------------------------------------------------------------------
    # Inférence sexe
    # --------------------------------------------------------------------------
    logger.info("\n=== [1/2] Inférence sexe ===")
    id2label_sex = {int(k): v for k, v in sex_cfg["id2label"].items()}
    df = predict_sex_for_documents(
        df, sex_model, tokenizer, device,
        id2label=id2label_sex,
        max_length=cfg["model"]["max_length"],
        truncation_strategy=sex_cfg.get("truncation_strategy", "head"),
    )

    # --------------------------------------------------------------------------
    # Inférence dates
    # --------------------------------------------------------------------------
    logger.info("\n=== [2/2] Inférence dates ===")
    df = predict_dates_for_documents(
        df, date_model, tokenizer, device,
        label2id=date_cfg["label2id"],
        context_window=date_cfg["context_window"],
        max_length=date_cfg["max_length"],
        consolidation_threshold=threshold,
    )

    # --------------------------------------------------------------------------
    # Vérification vs ground truth (si train set)
    # --------------------------------------------------------------------------
    if args.train_set and "sexe" in df.columns:
        logger.info("\n=== Vérification vs ground truth ===")
        # Sexe
        correct_sex = (df["sexe_pred"] == df["sexe"]).sum()
        valid_mask = df["sexe"].isin(["homme", "femme"])
        acc_sex = correct_sex / valid_mask.sum() if valid_mask.sum() > 0 else 0
        logger.info(f"Accuracy sexe (train) : {acc_sex:.4f} ({correct_sex}/{valid_mask.sum()})")

        # Dates
        from src.evaluate import evaluate_date_predictions
        doc_preds = df[["filename", "date_accident_pred", "date_consolidation_pred"]].rename(
            columns={"date_accident_pred": "date_accident",
                     "date_consolidation_pred": "date_consolidation"}
        ).to_dict("records")
        evaluate_date_predictions(df, doc_preds)

    # --------------------------------------------------------------------------
    # Génération de la soumission
    # --------------------------------------------------------------------------
    logger.info("\n=== Génération de la soumission ===")
    generate_submission(df, output_path)
    logger.info(f"\nSoumission prête : {output_path}")


if __name__ == "__main__":
    main()
