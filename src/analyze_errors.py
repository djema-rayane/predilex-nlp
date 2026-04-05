"""
analyze_errors.py
-----------------
Analyse des erreurs du modèle de dates sur le test set interne.

Lit le fichier predictions_dates_test_interne.csv et affiche :
  1. Résumé global des erreurs
  2. Cas où accident est mal prédit → affiche le texte du document
  3. Cas où consolidation est mal prédite → affiche le texte du document
  4. Patterns récurrents dans les erreurs

Usage (sur Colab) :
  python src/analyze_errors.py --config configs/config.yaml
  python src/analyze_errors.py --config configs/config.yaml --task accident
  python src/analyze_errors.py --config configs/config.yaml --task consolidation --n 5
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_text(filename, txt_dir):
    path = Path(txt_dir) / filename
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return ""


def print_separator():
    print("=" * 70)


def analyze_errors(preds_path, txt_dir, task="both", n_examples=3):
    df = pd.read_csv(preds_path)

    print_separator()
    print(f"ANALYSE DES ERREURS — {len(df)} documents dans le test interne")
    print_separator()

    # Résumé global
    acc_ok = df["accident_ok"].sum()
    con_ok = df["consolidation_ok"].sum()
    print(f"\ndate_accident    : {acc_ok}/{len(df)} corrects ({acc_ok/len(df)*100:.1f}%)")
    print(f"date_consolidation: {con_ok}/{len(df)} corrects ({con_ok/len(df)*100:.1f}%)")

    # Types d'erreurs accident
    acc_errors = df[~df["accident_ok"]]
    print(f"\n--- Erreurs accident ({len(acc_errors)} docs) ---")
    pred_nc = acc_errors[acc_errors["pred_accident"] == "n.c."]
    pred_wrong = acc_errors[acc_errors["pred_accident"] != "n.c."]
    print(f"  Prédit n.c. alors qu'il y a une vraie date : {len(pred_nc)}")
    print(f"  Prédit une mauvaise date                   : {len(pred_wrong)}")

    # Types d'erreurs consolidation
    con_errors = df[~df["consolidation_ok"]]
    print(f"\n--- Erreurs consolidation ({len(con_errors)} docs) ---")
    # Cas où vrai=n.c. mais on prédit une date
    false_pos = con_errors[
        (con_errors["true_consolidation"] == "n.c.") &
        (con_errors["pred_consolidation"] != "n.c.")
    ]
    # Cas où vrai=date mais on prédit n.c.
    false_neg = con_errors[
        (con_errors["true_consolidation"] != "n.c.") &
        (con_errors["pred_consolidation"] == "n.c.")
    ]
    # Cas où on prédit une mauvaise date
    wrong_date = con_errors[
        (con_errors["true_consolidation"] != "n.c.") &
        (con_errors["pred_consolidation"] != "n.c.")
    ]
    print(f"  Faux positifs (prédit date, vrai=n.c.)     : {len(false_pos)}")
    print(f"  Faux négatifs (prédit n.c., vrai=date)     : {len(false_neg)}")
    print(f"  Mauvaise date prédite                      : {len(wrong_date)}")

    # Exemples détaillés
    if task in ("accident", "both") and len(acc_errors) > 0:
        print_separator()
        print(f"\nEXEMPLES D'ERREURS ACCIDENT (max {n_examples})")
        print_separator()
        for _, row in acc_errors.head(n_examples).iterrows():
            print(f"\nFichier : {row['filename']}")
            print(f"  Vrai  : {row['true_accident']}")
            print(f"  Prédit: {row['pred_accident']}")
            # Chercher la vraie date dans le texte
            text = load_text(row["filename"], txt_dir)
            if text:
                true_date = str(row["true_accident"])
                # Chercher le contexte autour de la date
                # Format YYYY-MM-DD → chercher les composantes
                parts = true_date.split("-")
                if len(parts) == 3:
                    year = parts[0]
                    # Chercher les lignes contenant l'année
                    lines = [l.strip() for l in text.split("\n") if year in l and l.strip()]
                    print(f"  Lignes contenant {year} :")
                    for line in lines[:4]:
                        print(f"    >> {line[:120]}")

    if task in ("consolidation", "both") and len(con_errors) > 0:
        print_separator()
        print(f"\nEXEMPLES D'ERREURS CONSOLIDATION (max {n_examples})")
        print_separator()

        # Montrer surtout les faux négatifs (on rate des vraies dates)
        to_show = false_neg if len(false_neg) > 0 else con_errors
        for _, row in to_show.head(n_examples).iterrows():
            print(f"\nFichier : {row['filename']}")
            print(f"  Vrai  : {row['true_consolidation']}")
            print(f"  Prédit: {row['pred_consolidation']}")
            text = load_text(row["filename"], txt_dir)
            if text and row["true_consolidation"] != "n.c.":
                true_date = str(row["true_consolidation"])
                parts = true_date.split("-")
                if len(parts) == 3:
                    year = parts[0]
                    lines = [l.strip() for l in text.split("\n") if year in l and l.strip()]
                    # Chercher aussi "consolidation" dans le texte
                    consol_lines = [
                        l.strip() for l in text.split("\n")
                        if "consolidat" in l.lower() and l.strip()
                    ]
                    print(f"  Lignes contenant {year} :")
                    for line in lines[:3]:
                        print(f"    >> {line[:120]}")
                    print(f"  Lignes contenant 'consolidation' :")
                    for line in consol_lines[:3]:
                        print(f"    >> {line[:120]}")

    print_separator()
    print("\nConclusion : regarder les patterns ci-dessus pour identifier")
    print("les types de phrases que le modèle rate systématiquement.")
    print_separator()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--task", choices=["accident", "consolidation", "both"], default="both")
    parser.add_argument("--n", type=int, default=3, help="Nombre d'exemples à afficher")
    args = parser.parse_args()

    cfg = load_config(args.config)
    preds_path = Path(cfg["paths"]["processed_data_dir"]) / "predictions_dates_test_interne.csv"

    if not preds_path.exists():
        print(f"Fichier introuvable : {preds_path}")
        print("Lance d'abord : python src/train_dates.py")
        sys.exit(1)

    analyze_errors(
        preds_path=preds_path,
        txt_dir=cfg["paths"]["x_train_txt_dir"],
        task=args.task,
        n_examples=args.n,
    )


if __name__ == "__main__":
    main()
