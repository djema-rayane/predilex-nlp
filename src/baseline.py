"""
baseline.py
-----------
Calcule les scores des baselines pour comparaison avec notre modèle.

3 baselines pour les dates :
  1. Première date du document
  2. Date la plus fréquente dans le document
  3. Regex + mots-clés (consolidation/accident dans la phrase) — la plus forte

1 baseline pour le sexe :
  - Toujours prédire "homme" (classe majoritaire)

Usage :
  python src/baseline.py --config configs/config.yaml
"""

import argparse
import re
import sys
import os
from collections import Counter

sys.path.insert(0, os.getcwd())

from src.data_loader import load_config, load_train_data, split_train_val_test
from src.preprocessing import DATE_PATTERN, normalize_date, clean_text


ACCIDENT_KEYWORDS  = ["accident", "victime", "blessé", "blessée", "chute", "heurté", "écrasé", "percuté"]
CONSOLIDATION_KEYWORDS = ["consolidation", "consolidé", "consolidée", "fixée au", "fixé au", "guérison"]


def extract_all_dates(text):
    """Retourne liste de (date_normalisée, phrase) pour tout le texte."""
    results = []
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if s.strip()]
    for sent in sentences:
        for match in DATE_PATTERN.finditer(sent):
            dn = normalize_date(match)
            if dn:
                results.append((dn, sent))
    return results


def baseline_first_date(text):
    """Prend la première date trouvée dans le document."""
    dates = extract_all_dates(text)
    return dates[0][0] if dates else "n.c."


def baseline_most_frequent(text):
    """Prend la date la plus fréquente dans le document."""
    dates = extract_all_dates(text)
    if not dates:
        return "n.c."
    counts = Counter(d for d, _ in dates)
    return counts.most_common(1)[0][0]


def baseline_keywords(text):
    """
    Regex + mots-clés :
    - date_accident : phrase contenant un mot-clé accident + une date
    - date_consolidation : phrase contenant un mot-clé consolidation + une date
    Retourne (pred_accident, pred_consolidation)
    """
    dates = extract_all_dates(text)

    best_acc  = None
    best_cons = None

    for dn, sent in dates:
        sent_lower = sent.lower()
        if best_acc is None and any(kw in sent_lower for kw in ACCIDENT_KEYWORDS):
            best_acc = dn
        if best_cons is None and any(kw in sent_lower for kw in CONSOLIDATION_KEYWORDS):
            best_cons = dn

    return best_acc or "n.c.", best_cons or "n.c."


def evaluate(preds, trues, name):
    """Calcule l'accuracy globale, n.c. et dates."""
    NC = {"n.c.", "n.a.", "", "nan"}
    correct = total = nc_ok = nc_tot = date_ok = date_tot = 0
    for pred, true in zip(preds, trues):
        total += 1
        true_s = str(true).strip()
        pred_s = str(pred).strip()
        is_nc = true_s.lower() in NC

        if is_nc:
            nc_tot += 1
            if pred_s.lower() in NC:
                correct += 1
                nc_ok += 1
        else:
            date_tot += 1
            if pred_s == true_s:
                correct += 1
                date_ok += 1

    acc = correct / total if total else 0
    nc_acc = nc_ok / nc_tot if nc_tot else 1.0
    date_acc = date_ok / date_tot if date_tot else 0

    print(f"\n  {name} :")
    print(f"    Accuracy globale : {acc:.4f} ({correct}/{total})")
    print(f"    Accuracy n.c.    : {nc_acc:.4f} ({nc_ok}/{nc_tot})")
    print(f"    Accuracy dates   : {date_acc:.4f} ({date_ok}/{date_tot})")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_train_data(cfg)

    # Même split que le modèle (seed=42, test_split=0.10)
    date_cfg = cfg["date_classification"]
    _, _, test_df = split_train_val_test(
        df,
        val_split=date_cfg["val_split"],
        test_split=0.10,
        seed=date_cfg["seed"],
        label_col="sexe",
    )
    print(f"Test set : {len(test_df)} documents\n")

    # -------------------------------------------------------------------------
    # Baseline SEXE : toujours prédire homme
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("BASELINE SEXE — Toujours prédire 'homme'")
    print("=" * 60)
    sex_true = test_df["sexe"].tolist()
    sex_pred = ["homme"] * len(sex_true)
    correct_sex = sum(p == t for p, t in zip(sex_pred, sex_true))
    print(f"  Accuracy : {correct_sex/len(sex_true):.4f} ({correct_sex}/{len(sex_true)})")

    # -------------------------------------------------------------------------
    # Baselines DATES
    # -------------------------------------------------------------------------
    first_acc, first_cons = [], []
    freq_acc,  freq_cons  = [], []
    kw_acc,    kw_cons    = [], []

    for _, row in test_df.iterrows():
        text = clean_text(row["text"])

        first_acc.append(baseline_first_date(text))
        first_cons.append(baseline_first_date(text))

        freq_acc.append(baseline_most_frequent(text))
        freq_cons.append(baseline_most_frequent(text))

        pa, pc = baseline_keywords(text)
        kw_acc.append(pa)
        kw_cons.append(pc)

    true_acc  = test_df["date_accident"].tolist()
    true_cons = test_df["date_consolidation"].tolist()

    print("\n" + "=" * 60)
    print("BASELINE 1 — Première date du document")
    print("=" * 60)
    evaluate(first_acc,  true_acc,  "date_accident")
    evaluate(first_cons, true_cons, "date_consolidation")

    print("\n" + "=" * 60)
    print("BASELINE 2 — Date la plus fréquente")
    print("=" * 60)
    evaluate(freq_acc,  true_acc,  "date_accident")
    evaluate(freq_cons, true_cons, "date_consolidation")

    print("\n" + "=" * 60)
    print("BASELINE 3 — Regex + mots-clés")
    print("=" * 60)
    evaluate(kw_acc,  true_acc,  "date_accident")
    evaluate(kw_cons, true_cons, "date_consolidation")

    print("\n" + "=" * 60)
    print("RÉSUMÉ — Notre modèle CamemBERT (meilleur run)")
    print("=" * 60)
    print("\n  date_accident :")
    print("    Accuracy globale : 0.7792 (60/77)")
    print("    Accuracy n.c.    : 1.0000 (3/3)")
    print("    Accuracy dates   : 0.7703 (57/74)")
    print("\n  date_consolidation :")
    print("    Accuracy globale : 0.7403 (57/77)")
    print("    Accuracy n.c.    : 0.9697 (32/33)")
    print("    Accuracy dates   : 0.5682 (25/44)")


if __name__ == "__main__":
    main()
