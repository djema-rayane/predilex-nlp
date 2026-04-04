"""
train_sex.py
------------
Fine-tuning CamemBERT pour la classification du sexe de la victime.

Objectif : prédire si la victime de l'accident est un homme ou une femme.

Pipeline :
  1. Chargement des données (770 documents)
  2. Split stratifié train / val / test (au niveau document)
  3. Tokenisation + construction des SexDatasets PyTorch
  4. Fine-tuning CamemBERT avec :
       - Class weights pour compenser le déséquilibre 73% homme / 27% femme
       - Early stopping sur f1_weighted
       - Mixed precision (fp16) si GPU disponible
  5. Évaluation sur le test set interne
  6. Sauvegarde du meilleur modèle

Gestion du déséquilibre de classes (73/27) :
  On utilise un WeightedTrainer qui surcharge la méthode compute_loss
  du Trainer HuggingFace pour passer des poids de classe à la CrossEntropyLoss.
  Effet : une erreur sur "femme" (classe rare) est pénalisée ~2.7x plus
  qu'une erreur sur "homme" (classe majoritaire).
  Cela évite que le modèle devienne un simple "classifieur homme-par-défaut".

Stratégie de truncation :
  Tous les documents dépassent 512 tokens. Le signal sexe (Monsieur/Madame)
  est dans les premiers 3.7% du texte (médiane) → first 512 tokens suffisent
  dans 94% des cas. Option head_tail disponible dans config.yaml.

Usage :
  cd "NLP S2"
  python src/train_sex.py                           # entraînement standard
  python src/train_sex.py --debug                  # mode rapide (50 docs, 2 epochs)
  python src/train_sex.py --config configs/config.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    get_class_weights,
    load_config,
    load_train_data,
    split_train_val_test,
)
from src.evaluate import compute_metrics_binary, full_evaluation_report
from src.model_utils import (
    count_parameters,
    get_device,
    is_fp16_available,
    load_model,
    load_tokenizer,
    save_model,
)
from src.preprocessing import SexDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Trainer avec poids de classe
# =============================================================================

class WeightedTrainer(Trainer):
    """
    Trainer HuggingFace avec CrossEntropyLoss pondérée.

    Pourquoi sous-classer Trainer ?
      Le Trainer standard utilise une CrossEntropyLoss uniforme (toutes
      les classes ont le même poids). Avec un déséquilibre 73/27, cela
      conduit à un modèle qui prédit "homme" par défaut (~73% accuracy).

    Solution : surcharger compute_loss pour passer class_weights à la loss.
      weight_homme = 770 / (2 × 559) ≈ 0.69  (classe majoritaire, moins pénalisée)
      weight_femme = 770 / (2 × 206) ≈ 1.87  (classe minoritaire, plus pénalisée)

    Le modèle apprend ainsi à mieux distinguer les femmes, au prix d'une
    légère baisse de précision sur les hommes.
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Calcule la CrossEntropyLoss avec poids de classe.

        On extrait les labels des inputs, on fait le forward pass,
        puis on calcule la loss avec les poids de classe.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
        )
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# Préparation des labels
# =============================================================================

def prepare_sex_labels(df, label2id: dict) -> list:
    """
    Convertit la colonne 'sexe' en entiers via label2id.

    Gestion des cas n.c. :
      Les 5 documents avec sexe='n.c.' ont été mis dans train uniquement
      par split_train_val_test. Ici on les mappe à 0 (homme) par défaut
      (ils seront ignorés dans la loss grâce aux poids, ou on peut les exclure).

    Returns
    -------
    Liste d'entiers prête pour SexDataset
    """
    labels = []
    n_nc = 0
    for val in df["sexe"]:
        label_id = label2id.get(str(val).strip().lower())
        if label_id is None:
            n_nc += 1
            label_id = 0  # fallback : classe par défaut
        labels.append(int(label_id))

    if n_nc > 0:
        logger.warning(
            f"{n_nc} labels sexe non reconnus (n.c./NaN) → mappés à 0 (homme)"
        )
    return labels


# =============================================================================
# Main
# =============================================================================

def main():
    # --------------------------------------------------------------------------
    # Arguments CLI
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Fine-tuning CamemBERT — Classification du sexe"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Chemin vers le fichier de configuration YAML"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Mode debug : 50 documents, 2 epochs (test rapide du pipeline)"
    )
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------------------
    cfg = load_config(args.config)
    sex_cfg = cfg["sex_classification"]

    device = get_device()
    use_fp16 = sex_cfg["fp16"] and is_fp16_available()

    label2id = sex_cfg["label2id"]
    id2label = {int(k): v for k, v in sex_cfg["id2label"].items()}
    output_dir = sex_cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Device : {device} | FP16 : {use_fp16}")
    logger.info(f"Label mapping : {label2id}")

    # --------------------------------------------------------------------------
    # 1. Chargement des données
    # --------------------------------------------------------------------------
    logger.info("=== [1/7] Chargement des données ===")
    df = load_train_data(cfg)

    if args.debug:
        df = df.head(50)
        logger.info("MODE DEBUG : 50 documents, 2 epochs")

    # --------------------------------------------------------------------------
    # 2. Split train / val / test (stratifié sur 'sexe', niveau document)
    # --------------------------------------------------------------------------
    logger.info("=== [2/7] Split train/val/test ===")
    train_df, val_df, test_df = split_train_val_test(
        df,
        val_split=sex_cfg["val_split"],
        test_split=sex_cfg["test_split"],
        seed=sex_cfg["seed"],
        label_col="sexe",
    )

    # --------------------------------------------------------------------------
    # 3. Tokenisation + datasets PyTorch
    # --------------------------------------------------------------------------
    logger.info("=== [3/7] Tokenisation et datasets ===")
    tokenizer = load_tokenizer(cfg["model"]["name"])
    max_len = cfg["model"]["max_length"]
    trunc_strategy = sex_cfg.get("truncation_strategy", "head")

    train_labels = prepare_sex_labels(train_df, label2id)
    val_labels   = prepare_sex_labels(val_df, label2id)
    test_labels  = prepare_sex_labels(test_df, label2id)

    train_dataset = SexDataset(
        train_df["text"].tolist(), train_labels,
        tokenizer, max_len, truncation_strategy=trunc_strategy
    )
    val_dataset = SexDataset(
        val_df["text"].tolist(), val_labels,
        tokenizer, max_len, truncation_strategy=trunc_strategy
    )
    test_dataset = SexDataset(
        test_df["text"].tolist(), test_labels,
        tokenizer, max_len, truncation_strategy=trunc_strategy
    )

    logger.info(
        f"Datasets — Train: {len(train_dataset)} | "
        f"Val: {len(val_dataset)} | Test: {len(test_dataset)}"
    )

    # --------------------------------------------------------------------------
    # 4. Chargement du modèle + poids de classe
    # --------------------------------------------------------------------------
    logger.info("=== [4/7] Chargement du modèle ===")
    model = load_model(
        model_name=cfg["model"]["name"],
        num_labels=sex_cfg["num_labels"],
        label2id=label2id,
        id2label=id2label,
    )
    count_parameters(model)

    # Calcul des poids de classe sur le train set
    class_weight_list = get_class_weights(train_df, "sexe", label2id)
    class_weights = torch.tensor(class_weight_list, dtype=torch.float)
    logger.info(f"Poids de classe : homme={class_weight_list[0]:.3f}, femme={class_weight_list[1]:.3f}")

    # --------------------------------------------------------------------------
    # 5. Arguments d'entraînement
    # --------------------------------------------------------------------------
    logger.info("=== [5/7] Configuration de l'entraînement ===")
    epochs = 2 if args.debug else sex_cfg["num_epochs"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=sex_cfg["batch_size"],
        per_device_eval_batch_size=sex_cfg["batch_size"] * 2,
        learning_rate=sex_cfg["learning_rate"],
        warmup_ratio=sex_cfg["warmup_ratio"],
        weight_decay=sex_cfg["weight_decay"],

        # Évaluation à chaque epoch (pas après chaque step)
        eval_strategy="epoch",
        save_strategy="epoch",

        # Sauvegarder le meilleur modèle selon f1_weighted
        load_best_model_at_end=True,
        metric_for_best_model=sex_cfg.get("metric_for_best_model", "f1_weighted"),
        greater_is_better=True,

        fp16=use_fp16,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=cfg["logging"]["log_steps"],
        report_to=["wandb"] if cfg["logging"]["use_wandb"] else ["none"],
        seed=sex_cfg["seed"],

        # Windows : évite les problèmes de multiprocessing avec DataLoader
        dataloader_num_workers=0,

        # Ne pas sauvegarder tous les checkpoints (économise l'espace disque)
        save_total_limit=2,
    )

    logger.info(
        f"Entraînement : {epochs} epochs | "
        f"batch={sex_cfg['batch_size']} | "
        f"lr={sex_cfg['learning_rate']} | "
        f"warmup={sex_cfg['warmup_ratio']}"
    )

    # --------------------------------------------------------------------------
    # 6. Entraînement
    # --------------------------------------------------------------------------
    logger.info("=== [6/7] Entraînement ===")

    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=sex_cfg["early_stopping_patience"]
        )
    ]

    # WeightedTrainer pour la loss pondérée
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_binary,
        callbacks=callbacks,
    )

    if cfg["logging"]["use_wandb"]:
        import wandb
        wandb.init(
            project=cfg["logging"]["wandb_project"],
            name="sex_classification",
        )

    train_result = trainer.train()
    logger.info(f"Entraînement terminé : {train_result.metrics}")

    # --------------------------------------------------------------------------
    # 7. Évaluation finale sur le test set interne
    # --------------------------------------------------------------------------
    logger.info("=== [7/7] Évaluation finale sur le test set ===")

    test_results = trainer.predict(test_dataset)
    test_preds = np.argmax(test_results.predictions, axis=-1).tolist()

    full_evaluation_report(
        y_true=test_labels,
        y_pred=test_preds,
        label_names=list(label2id.keys()),
        task_name="Classification du Sexe — Test Set",
    )

    # --------------------------------------------------------------------------
    # Sauvegarde du meilleur modèle
    # --------------------------------------------------------------------------
    final_model_dir = os.path.join(output_dir, "best_model")
    save_model(trainer.model, tokenizer, final_model_dir)
    logger.info(f"Meilleur modèle sauvegardé dans : {final_model_dir}")
    logger.info("Pour lancer l'inférence : python src/predict.py")

    return trainer.model, tokenizer


if __name__ == "__main__":
    main()
