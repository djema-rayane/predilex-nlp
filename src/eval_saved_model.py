"""
eval_saved_model.py
-------------------
Évalue le modèle de dates sauvegardé sans réentraîner.

Charge le modèle depuis models/date_classifier/best_model,
recrée le même split test (seed fixe → reproductible),
et affiche les métriques niveau document.

Usage (sur Colab, ~2 min) :
  python src/eval_saved_model.py --config configs/config.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.getcwd())

from src.data_loader import load_config, load_train_data, split_train_val_test
from src.evaluate import evaluate_date_predictions, full_evaluation_report
from src.model_utils import get_device, is_fp16_available, load_tokenizer, load_trained_model
from src.preprocessing import DateDataset, clean_text, extract_date_sentences, label_date_sentences

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    date_cfg = cfg["date_classification"]

    device = get_device()
    label2id = date_cfg["label2id"]
    id2label = {int(k): v for k, v in date_cfg["id2label"].items()}
    context_window = date_cfg["context_window"]
    max_length = date_cfg["max_length"]
    threshold = date_cfg["threshold_consolidation"]
    threshold_acc = date_cfg.get("threshold_accident", 0.4)
    model_path = os.path.join(date_cfg["output_dir"], "best_model")

    if not Path(model_path).exists():
        logger.error(f"Modèle introuvable : {model_path}")
        logger.error("Lance d'abord : python src/train_dates.py")
        sys.exit(1)

    logger.info(f"Chargement du modèle depuis {model_path}")
    tokenizer = load_tokenizer(cfg["model"]["name"])
    model = load_trained_model(model_path, num_labels=date_cfg["num_labels"])
    model.to(device)
    model.eval()

    logger.info("Chargement des données + recréation du split test (même seed)")
    df = load_train_data(cfg)
    _, _, test_df = split_train_val_test(
        df,
        val_split=date_cfg["val_split"],
        test_split=0.10,
        seed=date_cfg["seed"],
        label_col="sexe",
    )
    logger.info(f"Test set : {len(test_df)} documents")

    # Extraction des phrases pour évaluation niveau phrase
    logger.info("Extraction des phrases candidates...")
    test_sent = label_date_sentences(test_df, context_window=context_window, label2id=label2id)
    test_dataset = DateDataset(
        test_sent["context"].tolist(), test_sent["label"].tolist(),
        tokenizer, max_length
    )

    # Évaluation niveau phrase
    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir="/tmp/eval_tmp",
        per_device_eval_batch_size=date_cfg["batch_size"] * 2,
        fp16=date_cfg["fp16"] and is_fp16_available(),
        report_to=["none"],
        dataloader_num_workers=0,
    )
    from src.evaluate import compute_metrics_multiclass
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics_multiclass,
    )
    test_results = trainer.predict(test_dataset)
    test_preds = np.argmax(test_results.predictions, axis=-1).tolist()
    test_labels = test_sent["label"].tolist()

    full_evaluation_report(
        y_true=test_labels,
        y_pred=test_preds,
        label_names=["date_accident", "date_consolidation", "autre_date"],
        task_name="Classification Dates — Niveau Phrase",
    )

    # Évaluation niveau document
    logger.info("--- Évaluation niveau DOCUMENT ---")

    from src.train_dates import predict_dates_for_document

    doc_predictions = []
    for _, row in test_df.iterrows():
        preds = predict_dates_for_document(
            row["text"], model, tokenizer, device,
            label2id, context_window, max_length, threshold, threshold_acc,
        )
        doc_predictions.append({
            "filename": row["filename"],
            "pred_accident": preds["date_accident"],
            "true_accident": row["date_accident"],
            "accident_ok": preds["date_accident"] == row["date_accident"],
            "pred_consolidation": preds["date_consolidation"],
            "true_consolidation": row["date_consolidation"],
            "consolidation_ok": preds["date_consolidation"] == row["date_consolidation"],
        })

    print("\n  === Précision au niveau document ===")
    evaluate_date_predictions(test_df, doc_predictions)

    # Sauvegarder le CSV d'erreurs
    preds_df = pd.DataFrame(doc_predictions)
    out_path = os.path.join(cfg["paths"]["processed_data_dir"], "predictions_dates_test_interne.csv")
    Path(cfg["paths"]["processed_data_dir"]).mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(out_path, index=False)
    logger.info(f"CSV sauvegarde : {out_path}")
    logger.info("Pour analyser les erreurs : python src/analyze_errors.py")


if __name__ == "__main__":
    main()
