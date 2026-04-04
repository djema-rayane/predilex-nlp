"""
evaluate.py
-----------
Fonctions d'évaluation : métriques, rapports, matrice de confusion.

Ce module fournit deux niveaux d'évaluation :

  1. NIVEAU PHRASE (pendant l'entraînement HuggingFace Trainer)
     → compute_metrics_binary    : sexe (accuracy, f1_weighted, f1_macro)
     → compute_metrics_multiclass: dates (accuracy, f1_weighted, f1_macro)

  2. NIVEAU DOCUMENT (logique métier réelle)
     → full_evaluation_report : rapport complet post-entraînement
     → evaluate_date_extraction : précision date_accident / date_consolidation
     → calibrate_consolidation_threshold : trouver le meilleur seuil pour n.c.

Pourquoi distinguer niveau phrase et niveau document pour les dates ?
  Le classifieur est entraîné au niveau phrase (3 classes).
  Mais l'objectif final est de prédire la BONNE date pour le document.
  La précision phrase-level peut être excellente mais si la logique de
  sélection (argmax + seuil) est mal calibrée, la précision document-level
  sera mauvaise → évaluer les deux est indispensable.

Métriques choisies :
  - Accuracy     : proportion de prédictions exactes
  - F1 Weighted  : F1 pondéré par le support de chaque classe
                   (tient compte du déséquilibre, utilisé pour comparer)
  - F1 Macro     : F1 moyen sur toutes les classes (même poids)
                   (pénalise fortement les erreurs sur les classes rares)
  - Pour les dates : F1 Macro est la métrique principale car les classes
    accident et consolidation sont rares face à "autre_date"
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from transformers import EvalPrediction

logger = logging.getLogger(__name__)


# =============================================================================
# Métriques pour HuggingFace Trainer (appelées à chaque epoch d'évaluation)
# =============================================================================

def compute_metrics_binary(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Métriques pour la classification binaire du sexe.

    Appelée automatiquement par le Trainer HuggingFace à chaque epoch
    sur le validation set. Retourne un dict de métriques.

    Métriques calculées :
      - accuracy    : (TP + TN) / total
      - f1_weighted : F1 pondéré par le support (métrique de suivi principale)
      - f1_macro    : F1 moyen non-pondéré (diagnostique du déséquilibre)

    Parameters
    ----------
    eval_pred : EvalPrediction contenant (predictions=logits, label_ids=labels)
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_m = f1_score(labels, preds, average="macro", zero_division=0)

    return {
        "accuracy":    round(float(acc), 4),
        "f1_weighted": round(float(f1_w), 4),
        "f1_macro":    round(float(f1_m), 4),
    }


def compute_metrics_multiclass(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Métriques pour la classification 3 classes des phrases de dates.

    Identique à compute_metrics_binary mais pour 3 classes.
    La métrique de suivi du Trainer est f1_macro (définie dans config.yaml)
    car elle pénalise les erreurs sur accident et consolidation (classes rares).

    Parameters
    ----------
    eval_pred : EvalPrediction contenant (predictions=logits, label_ids=labels)
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_m = f1_score(labels, preds, average="macro", zero_division=0)

    return {
        "accuracy":    round(float(acc), 4),
        "f1_weighted": round(float(f1_w), 4),
        "f1_macro":    round(float(f1_m), 4),
    }


# =============================================================================
# Rapport complet post-entraînement
# =============================================================================

def full_evaluation_report(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None,
    task_name: str = "Classification",
) -> Dict[str, float]:
    """
    Rapport d'évaluation complet avec matrice de confusion.

    Affiché une seule fois à la fin de l'entraînement sur le test set interne.
    Permet de diagnostiquer :
      - Quelles classes sont bien prédites
      - Les confusions systématiques (ex: consolidation confondue avec autre_date)
      - L'impact du déséquilibre de classes

    Parameters
    ----------
    y_true      : labels réels (entiers)
    y_pred      : labels prédits (entiers)
    label_names : noms des classes ["homme", "femme"] ou ["accident", "consol.", "autre"]
    task_name   : titre du rapport

    Returns
    -------
    Dict avec accuracy, f1_weighted, f1_macro
    """
    acc = accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  RAPPORT D'ÉVALUATION — {task_name}")
    print(sep)
    print(f"  Accuracy    : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 Weighted : {f1_w:.4f}")
    print(f"  F1 Macro    : {f1_m:.4f}")
    print()
    print(classification_report(
        y_true, y_pred,
        target_names=label_names,
        zero_division=0,
        digits=4,
    ))

    # Matrice de confusion avec en-têtes
    cm = confusion_matrix(y_true, y_pred)
    print("  Matrice de confusion :")
    if label_names:
        # En-tête colonnes (prédictions)
        col_width = max(len(n) for n in label_names) + 2
        header = " " * (col_width + 2) + "  ".join(
            f"{n:>{col_width}}" for n in label_names
        )
        print(f"  {'Réel \\ Prédit':<{col_width+2}}{header[col_width+2:]}")
        print(f"  {'-' * (col_width * (len(label_names)+1) + 4)}")
        for i, row_vals in enumerate(cm):
            row_label = label_names[i] if label_names else str(i)
            row_str = "  ".join(f"{v:>{col_width}}" for v in row_vals)
            print(f"  {row_label:>{col_width}}  {row_str}")
    else:
        for row_vals in cm:
            print("  " + "  ".join(f"{v:>6}" for v in row_vals))

    print(f"{sep}\n")

    return {
        "accuracy":    round(float(acc), 4),
        "f1_weighted": round(float(f1_w), 4),
        "f1_macro":    round(float(f1_m), 4),
    }


# =============================================================================
# Évaluation niveau document (logique métier)
# =============================================================================

def evaluate_date_predictions(
    test_df,
    doc_predictions: List[Dict],
) -> Dict[str, float]:
    """
    Évalue la précision des prédictions de dates au niveau DOCUMENT.

    C'est la métrique métier réelle : est-ce qu'on a trouvé la BONNE date
    dans le document, pas seulement est-ce qu'on classe bien les phrases.

    Gestion des cas n.c. / n.a. :
      - Si ground truth est n.c. et prédiction est n.c. → CORRECT
      - Si ground truth est n.c. et prédiction est une date → FAUX
      - Si ground truth est une date et prédiction est n.c. → FAUX
      - Si ground truth == prédiction (exact) → CORRECT

    Parameters
    ----------
    test_df         : DataFrame avec colonnes filename, date_accident, date_consolidation
    doc_predictions : liste de dicts {filename, date_accident, date_consolidation}

    Returns
    -------
    Dict avec accuracy par colonne et métriques globales
    """
    import pandas as pd
    preds_df = pd.DataFrame(doc_predictions)

    NC_VALUES = {"n.c.", "n.a.", "", "nan"}
    results = {}

    for col in ["date_accident", "date_consolidation"]:
        correct = 0
        total = len(test_df)
        nc_gt_correct = 0
        nc_gt_total = 0
        date_gt_correct = 0
        date_gt_total = 0

        for _, row in test_df.iterrows():
            pred_rows = preds_df[preds_df["filename"] == row["filename"]]
            if len(pred_rows) == 0:
                continue

            gt = str(row[col]).strip().lower()
            pred = str(pred_rows[col].values[0]).strip().lower()

            is_nc_gt = gt in NC_VALUES
            is_nc_pred = pred in NC_VALUES

            if is_nc_gt:
                nc_gt_total += 1
                if is_nc_pred:
                    correct += 1
                    nc_gt_correct += 1
            else:
                date_gt_total += 1
                # Comparaison case-insensitive, normalisation légère
                gt_norm = str(row[col]).strip()
                pred_norm = str(pred_rows[col].values[0]).strip()
                if gt_norm == pred_norm:
                    correct += 1
                    date_gt_correct += 1

        acc = correct / total if total > 0 else 0.0
        nc_acc = nc_gt_correct / nc_gt_total if nc_gt_total > 0 else 1.0
        date_acc = date_gt_correct / date_gt_total if date_gt_total > 0 else 0.0

        results[col] = {
            "accuracy_global": round(acc, 4),
            "accuracy_nc": round(nc_acc, 4),
            "accuracy_dates": round(date_acc, 4),
            "total": total,
            "nc_total": nc_gt_total,
            "date_total": date_gt_total,
        }

        print(f"\n  {col} :")
        print(f"    Accuracy globale : {acc:.4f} ({correct}/{total})")
        print(f"    Accuracy n.c.    : {nc_acc:.4f} ({nc_gt_correct}/{nc_gt_total})")
        print(f"    Accuracy dates   : {date_acc:.4f} ({date_gt_correct}/{date_gt_total})")

    return results


# =============================================================================
# Calibration du seuil de consolidation
# =============================================================================

def calibrate_consolidation_threshold(
    val_df,
    doc_predictions_val: List[Dict],
    proba_consolidation: List[float],
    thresholds: Optional[List[float]] = None,
) -> float:
    """
    Trouve le seuil optimal pour prédire n.c. vs date de consolidation.

    Problème : 42% des documents ont date_consolidation = n.c. ou n.a.
    Si le modèle prédit une date pour ces documents → erreur.
    Solution : si max_proba(consolidation) < seuil → prédire n.c.

    On teste plusieurs seuils sur le validation set et on choisit
    celui qui maximise l'accuracy globale de date_consolidation.

    Parameters
    ----------
    val_df                 : DataFrame de validation avec labels
    doc_predictions_val    : prédictions sans seuillage (probas brutes)
    proba_consolidation    : probabilité max de la classe consolidation par doc
    thresholds             : liste de seuils à tester (default: 0.1 à 0.9)

    Returns
    -------
    Seuil optimal (float entre 0 et 1)
    """
    if thresholds is None:
        thresholds = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]

    NC_VALUES = {"n.c.", "n.a.", ""}
    best_threshold = 0.5
    best_acc = 0.0

    print("\n  Calibration du seuil de consolidation :")
    print(f"  {'Seuil':>8} | {'Accuracy':>10} | {'n.c. correct':>14} | {'date correct':>14}")
    print(f"  {'-'*60}")

    for threshold in thresholds:
        correct = 0
        total = len(val_df)

        for i, (_, row) in enumerate(val_df.iterrows()):
            if i >= len(proba_consolidation):
                break

            gt = str(row.get("date_consolidation", "")).strip()
            is_nc_gt = gt.lower() in NC_VALUES or gt == ""

            proba = proba_consolidation[i] if i < len(proba_consolidation) else 0.0
            predicted_nc = proba < threshold

            if is_nc_gt and predicted_nc:
                correct += 1
            elif not is_nc_gt and not predicted_nc:
                # Date prédite + date GT : évaluation séparée
                correct += 1  # approximation (la date exacte est vérifiée ailleurs)

        acc = correct / total if total > 0 else 0.0

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

        print(f"  {threshold:>8.2f} | {acc:>10.4f}")

    print(f"\n  → Meilleur seuil : {best_threshold:.2f} (accuracy={best_acc:.4f})")
    return best_threshold


# =============================================================================
# Test rapide
# =============================================================================

if __name__ == "__main__":
    # Test sur des prédictions fictives
    print("=== Test evaluate.py ===\n")

    # Tâche sexe (binaire)
    y_true = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0]
    y_pred = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1]

    print("--- Classification binaire (sexe) ---")
    results = full_evaluation_report(
        y_true, y_pred,
        label_names=["homme", "femme"],
        task_name="Test Sexe",
    )
    print(f"Résultats : {results}\n")

    # Tâche dates (3 classes)
    y_true_dates = [0, 2, 2, 1, 2, 2, 0, 2, 1, 2]
    y_pred_dates = [0, 2, 2, 2, 2, 2, 0, 2, 1, 2]

    print("--- Classification 3 classes (dates) ---")
    full_evaluation_report(
        y_true_dates, y_pred_dates,
        label_names=["date_accident", "date_consolidation", "autre_date"],
        task_name="Test Dates",
    )
