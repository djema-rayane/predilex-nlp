"""
train_dates.py
--------------
Fine-tuning CamemBERT pour l'extraction des dates (accident + consolidation).

Pipeline hybride en deux étapes :
  ÉTAPE 1 — Extraction rule-based (NLP classique)
    Regex → toutes les phrases contenant une date sont extraites.
    Chaque phrase + son contexte (±2 phrases) forme un exemple.
    Ce contexte est crucial : "consolidation fixée au 7 nov." n'a de sens
    qu'avec la phrase précédente mentionnant "consolidation".

  ÉTAPE 2 — Classification neurale (Deep NLP)
    CamemBERT fine-tuné → 3 classes :
      0 = date_accident      : "le 9 avril 1991, M. X... a été victime..."
      1 = date_consolidation : "sa consolidation était fixée au 7 nov 1982..."
      2 = autre_date         : date de jugement, rapport médical, etc.

  LOGIQUE MÉTIER (inférence document)
    Pour chaque document :
      - Retenir la phrase accident avec le score max  → date_accident
      - Retenir la phrase consolidation si score > seuil sinon → "n.c."

Défi principal — Déséquilibre extrême :
  Sur ~25 000 phrases candidates :
    ~770 accident (3%)  |  ~450 consolidation (2%)  |  ~23 780 autre (95%)
  Solution : oversampling des classes rares avant l'entraînement.
  Effet : le modèle n'apprend pas à tout prédire "autre_date".

Usage :
  cd "NLP S2"
  python src/train_dates.py                         # entraînement complet
  python src/train_dates.py --debug                # mode rapide (30 docs)
  python src/train_dates.py --predict-only         # inférence uniquement
  python src/train_dates.py --calibrate            # recalibrer le seuil seulement
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_config, load_test_data, load_train_data, split_train_val_test
from src.evaluate import (
    compute_metrics_multiclass,
    evaluate_date_predictions,
    full_evaluation_report,
)
from src.model_utils import (
    count_parameters,
    get_device,
    is_fp16_available,
    load_model,
    load_tokenizer,
    load_trained_model,
    save_model,
)
from src.preprocessing import (
    DateDataset,
    clean_text,
    extract_date_sentences,
    label_date_sentences,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Oversampling des classes rares
# =============================================================================

def oversample_minority_classes(
    df: pd.DataFrame,
    label_col: str = "label",
    target_ratio: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sur-échantillonne les classes minoritaires (accident, consolidation).

    Problème : avec ~3% accident et ~2% consolidation, un réseau de neurones
    non-équilibré apprendra à tout prédire "autre_date" (~95% accuracy !).
    C'est le paradoxe de l'accuracy avec classes déséquilibrées.

    Stratégie :
      Pour chaque classe rare dont le count < target_ratio × count_majoritaire,
      on duplique les exemples jusqu'à atteindre ce ratio.

      Exemple : 800 "autre", 20 "accident" (ratio=0.3)
        → target = 0.3 × 800 = 240 exemples accident
        → on duplique les 20 exemples accident 12 fois → 240 exemples
        → résultat : 800 autre | 240 accident | 240 consolidation

    Alternative non-utilisée ici : SMOTE (génère de nouveaux exemples par
    interpolation dans l'espace des features). Mais SMOTE s'applique sur
    des features numériques, pas directement sur du texte brut.

    Parameters
    ----------
    df           : DataFrame avec colonne label
    label_col    : nom de la colonne de labels
    target_ratio : ratio minimum des classes rares / classe majoritaire
    seed         : reproductibilité

    Returns
    -------
    DataFrame rééquilibré, mélangé aléatoirement
    """
    class_counts = df[label_col].value_counts()
    majority_count = class_counts.max()
    target_count = int(majority_count * target_ratio)

    frames = [df]
    for cls, count in class_counts.items():
        if count < target_count:
            minority_df = df[df[label_col] == cls]
            needed = target_count - count
            # Répétition + sampling sans replacement jusqu'à needed
            n_repeats = (needed // count) + 1
            pool = pd.concat([minority_df] * n_repeats, ignore_index=True)
            extra = pool.sample(needed, random_state=seed)
            frames.append(extra)
            logger.info(
                f"Oversampling classe '{cls}': {count} → {count + needed} exemples "
                f"(+{needed} dupliqués)"
            )

    result = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=seed)
    logger.info(
        f"Distribution après oversampling : "
        f"{result[label_col].value_counts().sort_index().to_dict()}"
    )
    return result


# =============================================================================
# Inférence au niveau document
# =============================================================================

def predict_dates_for_document(
    text: str,
    model,
    tokenizer,
    device: torch.device,
    label2id: dict,
    context_window: int = 2,
    max_length: int = 256,
    consolidation_threshold: float = 0.4,
) -> dict:
    """
    Pour un document, prédit date_accident et date_consolidation.

    Algorithme :
      1. Extraire toutes les phrases candidates (regex)
      2. Classifier chaque phrase (CamemBERT → 3 classes)
      3. date_accident      = date de la phrase avec max proba_accident
      4. date_consolidation = date de la phrase avec max proba_consolidation,
                             si max_proba > threshold, sinon "n.c."

    Pourquoi un seuil pour la consolidation mais pas pour l'accident ?
      La consolidation est n.c. dans 42% des cas → le modèle doit pouvoir
      "décider" qu'il n'y en a pas. Sans seuil, il prédirait toujours une date.
      L'accident est presque toujours présent (4% de n.c.) → pas de seuil.

    Parameters
    ----------
    text                   : texte brut du document
    model                  : CamemBERT fine-tuné
    tokenizer              : tokenizer CamemBERT
    device                 : cuda ou cpu
    label2id               : mapping label → id
    context_window         : fenêtre de contexte (identique à l'entraînement)
    max_length             : longueur max en tokens
    consolidation_threshold: seuil de confiance pour la consolidation

    Returns
    -------
    {
        "date_accident":      "YYYY-MM-DD" ou "n.c.",
        "date_consolidation": "YYYY-MM-DD" ou "n.c.",
    }
    """
    date_sentences = extract_date_sentences(clean_text(text), context_window=context_window)

    if not date_sentences:
        return {"date_accident": "n.c.", "date_consolidation": "n.c."}

    contexts = [ds["context"] for ds in date_sentences]
    dates = [ds["date_normalized"] for ds in date_sentences]

    # Forward pass sur toutes les phrases candidates
    model.eval()
    inputs = tokenizer(
        contexts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits           # (n_phrases, 3)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    acc_id  = label2id["date_accident"]
    cons_id = label2id["date_consolidation"]

    # Meilleure date_accident (toujours prédire quelque chose)
    acc_scores = probs[:, acc_id]
    best_acc_idx = int(np.argmax(acc_scores))
    predicted_accident = dates[best_acc_idx] if dates[best_acc_idx] else "n.c."

    # Meilleure date_consolidation (avec seuil)
    cons_scores = probs[:, cons_id]
    best_cons_idx = int(np.argmax(cons_scores))
    best_cons_score = float(cons_scores[best_cons_idx])

    if best_cons_score >= consolidation_threshold and dates[best_cons_idx]:
        predicted_consolidation = dates[best_cons_idx]
    else:
        predicted_consolidation = "n.c."

    return {
        "date_accident": predicted_accident,
        "date_consolidation": predicted_consolidation,
        "_acc_score": float(acc_scores[best_acc_idx]),
        "_cons_score": best_cons_score,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    # --------------------------------------------------------------------------
    # Arguments CLI
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Fine-tuning CamemBERT — Classification des dates"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Chemin vers le fichier de configuration YAML"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Mode debug : 30 documents, 2 epochs"
    )
    parser.add_argument(
        "--predict-only", action="store_true",
        help="Inférence uniquement (charge le modèle sauvegardé)"
    )
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------
    cfg = load_config(args.config)
    date_cfg = cfg["date_classification"]

    device = get_device()
    use_fp16 = date_cfg["fp16"] and is_fp16_available()

    label2id = date_cfg["label2id"]
    id2label = {int(k): v for k, v in date_cfg["id2label"].items()}
    context_window = date_cfg["context_window"]
    max_length = date_cfg["max_length"]
    output_dir = date_cfg["output_dir"]
    threshold = date_cfg["threshold_consolidation"]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(cfg["model"]["name"])

    # --------------------------------------------------------------------------
    # Mode inférence uniquement (--predict-only)
    # --------------------------------------------------------------------------
    if args.predict_only:
        logger.info("=== Mode inférence uniquement ===")
        model_path = os.path.join(output_dir, "best_model")
        model = load_trained_model(model_path, num_labels=date_cfg["num_labels"])
        model.to(device)

        test_df = load_test_data(cfg)
        results = []
        for _, row in test_df.iterrows():
            preds = predict_dates_for_document(
                row["text"], model, tokenizer, device,
                label2id, context_window, max_length, threshold
            )
            results.append({
                "filename": row["filename"],
                "date_accident": preds["date_accident"],
                "date_consolidation": preds["date_consolidation"],
            })

        results_df = pd.DataFrame(results)
        out_path = os.path.join(
            cfg["paths"]["processed_data_dir"], "predictions_dates.csv"
        )
        Path(cfg["paths"]["processed_data_dir"]).mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path, index=False)
        logger.info(f"Prédictions sauvegardées : {out_path}")
        return

    # --------------------------------------------------------------------------
    # 1. Chargement des données
    # --------------------------------------------------------------------------
    logger.info("=== [1/8] Chargement des données ===")
    df = load_train_data(cfg)

    if args.debug:
        df = df.head(30)
        logger.info("MODE DEBUG : 30 documents, 2 epochs")

    # --------------------------------------------------------------------------
    # 2. Split au niveau document (AVANT labellisation pour éviter data leakage)
    # --------------------------------------------------------------------------
    logger.info("=== [2/8] Split train/val/test au niveau document ===")
    # On stratifie sur 'sexe' et non 'date_consolidation' car les labels de date
    # sont trop déséquilibrés (42% n.c.) pour être utilisés comme strate
    train_df, val_df, test_df = split_train_val_test(
        df,
        val_split=date_cfg["val_split"],
        test_split=0.10,
        seed=date_cfg["seed"],
        label_col="sexe",
    )

    # --------------------------------------------------------------------------
    # 3. Extraction + labellisation des phrases (APRÈS le split !)
    # --------------------------------------------------------------------------
    logger.info("=== [3/8] Extraction des phrases dates ===")
    train_sent = label_date_sentences(train_df, context_window=context_window, label2id=label2id)
    val_sent   = label_date_sentences(val_df, context_window=context_window, label2id=label2id)
    test_sent  = label_date_sentences(test_df, context_window=context_window, label2id=label2id)

    logger.info(
        f"Phrases — Train: {len(train_sent)} | Val: {len(val_sent)} | Test: {len(test_sent)}"
    )

    if len(train_sent) == 0:
        logger.error("Aucune phrase de date trouvée ! Vérifiez les fichiers texte.")
        return

    # --------------------------------------------------------------------------
    # 4. Oversampling des classes rares dans le train set
    # --------------------------------------------------------------------------
    logger.info("=== [4/8] Oversampling des classes rares ===")
    train_sent = oversample_minority_classes(
        train_sent,
        label_col="label",
        target_ratio=date_cfg["oversample_ratio"],
        seed=date_cfg["seed"],
    )

    # --------------------------------------------------------------------------
    # 5. Datasets PyTorch
    # --------------------------------------------------------------------------
    logger.info("=== [5/8] Construction des datasets ===")
    train_dataset = DateDataset(
        train_sent["context"].tolist(), train_sent["label"].tolist(),
        tokenizer, max_length
    )
    val_dataset = DateDataset(
        val_sent["context"].tolist(), val_sent["label"].tolist(),
        tokenizer, max_length
    )
    test_dataset = DateDataset(
        test_sent["context"].tolist(), test_sent["label"].tolist(),
        tokenizer, max_length
    )

    # --------------------------------------------------------------------------
    # 6. Modèle + arguments d'entraînement
    # --------------------------------------------------------------------------
    logger.info("=== [6/8] Chargement du modèle ===")
    model = load_model(
        model_name=cfg["model"]["name"],
        num_labels=date_cfg["num_labels"],
        label2id=label2id,
        id2label=id2label,
    )
    count_parameters(model)

    epochs = 2 if args.debug else date_cfg["num_epochs"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=date_cfg["batch_size"],
        per_device_eval_batch_size=date_cfg["batch_size"] * 2,
        learning_rate=date_cfg["learning_rate"],
        warmup_ratio=date_cfg["warmup_ratio"],
        weight_decay=date_cfg["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # F1 Macro : pénalise les erreurs sur les classes rares (accident, consolidation)
        metric_for_best_model=date_cfg.get("metric_for_best_model", "f1_macro"),
        greater_is_better=True,
        fp16=use_fp16,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=cfg["logging"]["log_steps"],
        report_to=["wandb"] if cfg["logging"]["use_wandb"] else ["none"],
        seed=date_cfg["seed"],
        dataloader_num_workers=0,
        save_total_limit=2,
    )

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=date_cfg["early_stopping_patience"]
        )
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_multiclass,
        callbacks=callbacks,
    )

    # --------------------------------------------------------------------------
    # 7. Entraînement
    # --------------------------------------------------------------------------
    logger.info("=== [7/8] Entraînement ===")
    train_result = trainer.train()
    logger.info(f"Entraînement terminé : {train_result.metrics}")

    # --------------------------------------------------------------------------
    # 8. Évaluation
    # --------------------------------------------------------------------------
    logger.info("=== [8/8] Évaluation ===")

    # --- 8a. Évaluation niveau PHRASE (sur le test set de phrases) ---
    logger.info("--- Évaluation niveau PHRASE ---")
    test_results = trainer.predict(test_dataset)
    test_preds_phrases = np.argmax(test_results.predictions, axis=-1).tolist()
    test_labels_phrases = test_sent["label"].tolist()

    full_evaluation_report(
        y_true=test_labels_phrases,
        y_pred=test_preds_phrases,
        label_names=["date_accident", "date_consolidation", "autre_date"],
        task_name="Classification Dates — Niveau Phrase",
    )

    # --- 8b. Évaluation niveau DOCUMENT (logique métier réelle) ---
    logger.info("--- Évaluation niveau DOCUMENT ---")
    model_for_inference = trainer.model
    model_for_inference.to(device)

    doc_predictions = []
    for _, row in test_df.iterrows():
        preds = predict_dates_for_document(
            row["text"], model_for_inference, tokenizer, device,
            label2id, context_window, max_length, threshold,
        )
        doc_predictions.append({
            "filename": row["filename"],
            "date_accident": preds["date_accident"],
            "date_consolidation": preds["date_consolidation"],
        })

    print("\n  === Précision au niveau document ===")
    evaluate_date_predictions(test_df, doc_predictions)

    # --------------------------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------------------------
    final_model_dir = os.path.join(output_dir, "best_model")
    save_model(trainer.model, tokenizer, final_model_dir)
    logger.info(f"Meilleur modèle sauvegardé : {final_model_dir}")

    # Sauvegarder aussi les prédictions du test set interne
    preds_df = pd.DataFrame(doc_predictions)
    preds_path = os.path.join(
        cfg["paths"]["processed_data_dir"], "predictions_dates_test_interne.csv"
    )
    Path(cfg["paths"]["processed_data_dir"]).mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(preds_path, index=False)
    logger.info(f"Prédictions test interne sauvegardées : {preds_path}")
    logger.info("Pour l'inférence finale : python src/predict.py")

    return trainer.model, tokenizer


if __name__ == "__main__":
    main()
