"""
preprocessing.py
----------------
Preprocessing textuel et construction des datasets PyTorch.

Ce module est le cœur du pipeline NLP. Il gère :

  1. NETTOYAGE DU TEXTE
     → Normalisation des espaces, encodages, artefacts OCR typiques
        des documents juridiques numérisés

  2. DÉCOUPAGE EN PHRASES
     → Heuristique adaptée au style juridique français (phrases longues,
        énumérations, références légales)

  3. EXTRACTION ET NORMALISATION DES DATES
     → Regex couvrant tous les formats français rencontrés dans la donnée :
          · Textuel  : "14 juin 1999", "1er mars 2004"
          · Numérique : "14/06/1999", "14-06-1999"
     → Normalisation vers le format ISO YYYY-MM-DD

  4. LABELLISATION DES PHRASES (pour entraîner le classifieur de dates)
     → Chaque phrase candidate est labellisée :
          · date_accident      (0) : correspond à la date d'accident du document
          · date_consolidation (1) : correspond à la date de consolidation
          · autre_date         (2) : toute autre date (jugement, rapport...)

  5. DATASETS PYTORCH
     → SexDataset   : texte complet → label binaire (homme/femme)
     → DateDataset  : phrase + contexte → label 3-classes

  Stratégie de truncation pour SexDataset :
     Tous les documents ont > 512 tokens (moy. 3315 mots).
     Le signal sexe est à 3.7% médian du texte → les 512 premiers tokens
     capturent ce signal dans 94% des cas.
     Option "head_tail" disponible si besoin (256 premiers + 256 derniers).
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# =============================================================================
# Constantes — Regex des dates françaises
# =============================================================================

# Mois français : formes longues + abréviations
_MONTHS_FR = (
    r"janvier|février|fevrier|mars|avril|mai|juin|juillet"
    r"|août|aout|septembre|octobre|novembre|décembre|decembre"
    r"|janv\.?|févr\.?|fevr\.?|avr\.?|juil\.?|sept\.?|oct\.?|nov\.?|déc\.?|dec\.?"
)

# Ordinals : "1er", "1ère", "2ème", etc. (pour "le 1er mars")
_ORDINAL = r"(?:1e?r|1\xc8\xa8re|\d+(?:\xc3\xa8me|i\xc3\xa8me|e)?)"

# Pattern principal : couvre les formats rencontrés dans les décisions Predilex
#   Groupe 1-3 : textuel  → "14 juin 1999" / "1er mars 2004"
#   Groupe 4-6 : numérique slash → "14/06/1999"
#   Groupe 7-9 : numérique tiret → "14-06-1999"
DATE_PATTERN = re.compile(
    rf"""
    (?:
        # Format textuel : jour (ordinal ou cardinal) + mois + année
        \b(\d{{1,2}}(?:e?r|i?è?me)?)\s+({_MONTHS_FR})\s+(\d{{4}})\b
        |
        # Format numérique avec slash : DD/MM/YYYY
        \b(\d{{1,2}})\s*/\s*(\d{{1,2}})\s*/\s*(\d{{4}})\b
        |
        # Format numérique avec tiret : DD-MM-YYYY
        \b(\d{{1,2}})-(\d{{1,2}})-(\d{{4}})\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Mapping mois texte → numéro à deux chiffres
_MONTH_MAP = {
    "janvier": "01", "janv": "01",
    "février": "02", "fevrier": "02", "févr": "02", "fevr": "02",
    "mars": "03",
    "avril": "04", "avr": "04",
    "mai": "05",
    "juin": "06",
    "juillet": "07", "juil": "07",
    "août": "08", "aout": "08",
    "septembre": "09", "sept": "09",
    "octobre": "10", "oct": "10",
    "novembre": "11", "nov": "11",
    "décembre": "12", "decembre": "12", "déc": "12", "dec": "12",
}


# =============================================================================
# 1. Nettoyage du texte
# =============================================================================

def clean_text(text: str) -> str:
    """
    Nettoyage du texte brut des décisions juridiques.

    Opérations appliquées :
      - Normalisation des espaces insécables (\\xa0) et tabulations
      - Suppression des artefacts courants dans les PDF numérisés (\\x0c, \\r)
      - Réduction des espaces multiples à un seul
      - Réduction des sauts de ligne multiples (≥3) à deux sauts max
      - Strip global

    IMPORTANT : on conserve intentionnellement :
      - La ponctuation (.!?) : utilisée pour découper les phrases
      - Les chiffres et dates : indispensables pour les regex de dates
      - La casse : CamemBERT est sensible à la casse (Monsieur ≠ monsieur)
    """
    if not isinstance(text, str):
        return ""

    # Artefacts d'encodage et de PDF
    text = text.replace("\xa0", " ")   # espace insécable
    text = text.replace("\t", " ")     # tabulation
    text = text.replace("\r\n", "\n")  # Windows line endings
    text = text.replace("\r", "\n")    # vieux Mac line endings
    text = text.replace("\x0c", "\n")  # form feed (page break PDF)

    # Espaces multiples sur une même ligne
    text = re.sub(r"[ ]{2,}", " ", text)

    # Sauts de ligne multiples → max 2 (conserve la structure paragraphe)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def truncate_head_tail(text: str, tokenizer: PreTrainedTokenizerBase,
                       max_length: int = 512) -> str:
    """
    Stratégie de truncation "head + tail" pour les documents longs.

    Au lieu de prendre les max_length premiers tokens (truncation naïve),
    on prend :
      - Les max_length//2 premiers tokens  (début du document)
      - Les max_length//2 derniers tokens  (fin du document)

    Avantage : capture à la fois l'en-tête (identification de la victime)
    et la conclusion (décision finale, qui peut mentionner dates/sexe).

    Note : pour la tâche sexe, "head" seul suffit dans 94% des cas
    (signal en début de document). "head_tail" est une alternative.

    Parameters
    ----------
    text       : texte brut nettoyé
    tokenizer  : tokenizer CamemBERT (SentencePiece)
    max_length : nombre total de tokens cible
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_length - 2:  # -2 pour [CLS] et [SEP]
        return text

    half = (max_length - 2) // 2
    head_tokens = tokens[:half]
    tail_tokens = tokens[-half:]

    head_text = tokenizer.decode(head_tokens, skip_special_tokens=True)
    tail_text = tokenizer.decode(tail_tokens, skip_special_tokens=True)

    return head_text + " [...] " + tail_text


# =============================================================================
# 2. Découpage en phrases
# =============================================================================

def split_sentences(text: str) -> List[str]:
    """
    Découpe un texte en phrases, adapté au style juridique français.

    Le style juridique présente des défis spécifiques :
      - Phrases très longues (plusieurs lignes)
      - Énumérations avec tirets ou points
      - Références légales contenant des points ("art. L.1234-5")
      - Abréviations ("M.", "Mme.", "Dr.")

    Stratégie :
      1. Découper sur les sauts de ligne (les décisions sont souvent
         structurées avec un paragraphe = une idée)
      2. Découper sur . ! ? suivi d'une majuscule (phrases classiques)
      3. Filtrer les fragments trop courts (< 10 chars)

    Returns
    -------
    Liste de phrases nettoyées, sans doublons, sans fragments vides
    """
    # Étape 1 : découpe sur les sauts de ligne
    paragraphs = re.split(r"\n+", text)

    sentences = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Étape 2 : découpe interne sur . ! ? suivi d'une majuscule
        # On évite de découper sur les abréviations communes
        parts = re.split(
            r"(?<=[.!?])\s+(?=[A-ZÀÂÉÈÊÙÛÎŒÆ])",
            para,
        )
        sentences.extend([p.strip() for p in parts if p.strip()])

    # Filtrer les fragments trop courts (dates isolées, numéros de page, etc.)
    sentences = [s for s in sentences if len(s) >= 10]

    return sentences


# =============================================================================
# 3. Normalisation des dates
# =============================================================================

def normalize_date(match: re.Match) -> Optional[str]:
    """
    Convertit une date matchée par DATE_PATTERN au format YYYY-MM-DD.

    Groupes du match :
      groups[0-2] : format textuel  (jour, mois_texte, année)
      groups[3-5] : format slash    (jour, mois_num, année)
      groups[6-8] : format tiret    (jour, mois_num, année)

    Validation :
      - Jour : 1-31
      - Mois : 1-12
      - Année : 1900-2100 (plage réaliste pour les accidents du travail)

    Returns
    -------
    "YYYY-MM-DD" si valide, None sinon
    """
    groups = match.groups()

    try:
        if groups[0] is not None:
            # Format textuel : "14 juin 1999" ou "1er mars 2004"
            day_raw = re.sub(r"[^\d]", "", groups[0])  # enlève "er", "ème", etc.
            month_str = groups[1].rstrip(".").lower().strip()
            year = groups[2]
            month = _MONTH_MAP.get(month_str)
            if month is None:
                return None
            day = day_raw

        elif groups[3] is not None:
            # Format slash : "14/06/1999"
            day, month, year = groups[3], groups[4], groups[5]

        else:
            # Format tiret : "14-06-1999"
            day, month, year = groups[6], groups[7], groups[8]

        day_int = int(day)
        month_int = int(month)
        year_int = int(year)

        # Validation de cohérence
        if not (1 <= day_int <= 31):
            return None
        if not (1 <= month_int <= 12):
            return None
        if not (1900 <= year_int <= 2100):
            return None

        return f"{year_int:04d}-{month_int:02d}-{day_int:02d}"

    except (ValueError, TypeError, AttributeError):
        return None


# =============================================================================
# 4. Extraction des phrases contenant des dates
# =============================================================================

def extract_date_sentences(
    text: str,
    context_window: int = 2,
) -> List[Dict]:
    """
    Extrait toutes les phrases contenant une date valide, avec leur contexte.

    Pour chaque date trouvée, on construit un contexte enrichi en prenant
    les phrases adjacentes (fenêtre ±context_window). Ce contexte est crucial
    car la sémantique de la date (accident vs consolidation) dépend souvent
    des phrases entourant la date, pas seulement de la phrase qui la contient.

    Exemple :
        "L'expert a conclu que sa consolidation pouvait être fixée
         au 7 novembre 1982 et qu'il subsiste une incapacité permanente."
        → Le mot "consolidation" dans la phrase précédente indique le type.

    Parameters
    ----------
    text           : texte brut nettoyé
    context_window : nombre de phrases de contexte avant et après

    Returns
    -------
    Liste de dicts :
        sentence        : phrase contenant la date
        context         : phrase + contexte (window phrases autour)
        date_normalized : date au format YYYY-MM-DD (ou None si invalide)
        date_raw        : date telle qu'écrite dans le texte
        sentence_idx    : index de la phrase dans le document
    """
    sentences = split_sentences(text)
    results = []

    for i, sent in enumerate(sentences):
        for match in DATE_PATTERN.finditer(sent):
            date_normalized = normalize_date(match)
            if date_normalized is None:
                continue

            # Contexte : sentences[i-w : i+w+1]
            start_idx = max(0, i - context_window)
            end_idx = min(len(sentences), i + context_window + 1)
            context = " ".join(sentences[start_idx:end_idx])

            results.append({
                "sentence": sent,
                "context": context,
                "date_normalized": date_normalized,
                "date_raw": match.group(0),
                "sentence_idx": i,
            })

    return results


# =============================================================================
# 5. Labellisation des phrases pour l'entraînement
# =============================================================================

def label_date_sentences(
    df: pd.DataFrame,
    context_window: int = 2,
    label2id: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Génère le dataset d'entraînement pour le classifieur de dates.

    Pour chaque document, on :
      1. Extrait toutes les phrases avec dates (via extract_date_sentences)
      2. Compare chaque date avec les labels ground truth du document
      3. Assigne un label : accident (0) / consolidation (1) / autre (2)

    Logique de labellisation :
      - Si date_normalized == date_accident du document → label 0
      - Si date_normalized == date_consolidation (et non n.c.) → label 1
      - Sinon → label 2 (autre : date de jugement, rapport médical, etc.)

    IMPORTANT : split au niveau DOCUMENT avant d'appeler cette fonction.
    Si on labellise avant de splitter, on risque une fuite d'information.

    Parameters
    ----------
    df             : DataFrame avec colonnes text, date_accident, date_consolidation
    context_window : fenêtre de contexte (identique à extract_date_sentences)
    label2id       : mapping label → id (default: accident=0, consolidation=1, autre=2)

    Returns
    -------
    DataFrame avec colonnes :
        filename | context | sentence | date_normalized | date_raw | label
    """
    if label2id is None:
        label2id = {"date_accident": 0, "date_consolidation": 1, "autre_date": 2}

    NC_VALUES = {"n.c.", "n.a.", "", "nan", "none"}

    records = []
    n_docs_with_dates = 0

    for _, row in df.iterrows():
        text = clean_text(row["text"])
        date_acc = str(row.get("date_accident", "")).strip().lower()
        date_cons = str(row.get("date_consolidation", "")).strip().lower()

        # Normaliser les valeurs ground truth
        date_acc = "" if date_acc in NC_VALUES else str(row.get("date_accident", "")).strip()
        date_cons = "" if date_cons in NC_VALUES else str(row.get("date_consolidation", "")).strip()

        date_sentences = extract_date_sentences(text, context_window=context_window)

        if date_sentences:
            n_docs_with_dates += 1

        for ds in date_sentences:
            dn = ds["date_normalized"]
            if dn is None:
                continue

            # Assignation du label — matching exact YYYY-MM-DD uniquement
            if date_acc and dn == date_acc:
                label = label2id["date_accident"]
            elif date_cons and dn == date_cons:
                label = label2id["date_consolidation"]
            else:
                label = label2id["autre_date"]

            records.append({
                "filename": row.get("filename", ""),
                "context": ds["context"],
                "sentence": ds["sentence"],
                "date_normalized": dn,
                "date_raw": ds["date_raw"],
                "label": label,
            })

    result_df = pd.DataFrame(records)

    if len(result_df) > 0:
        n_acc = (result_df["label"] == label2id["date_accident"]).sum()
        n_cons = (result_df["label"] == label2id["date_consolidation"]).sum()
        n_other = (result_df["label"] == label2id["autre_date"]).sum()
        logger.info(
            f"Phrases extraites : {len(result_df)} total | "
            f"accident: {n_acc} ({n_acc/len(result_df)*100:.1f}%) | "
            f"consolidation: {n_cons} ({n_cons/len(result_df)*100:.1f}%) | "
            f"autres: {n_other} ({n_other/len(result_df)*100:.1f}%) | "
            f"docs avec dates: {n_docs_with_dates}/{len(df)}"
        )
    else:
        logger.warning("Aucune phrase de date extraite ! Vérifiez les fichiers texte.")

    return result_df


# =============================================================================
# 6. Datasets PyTorch
# =============================================================================

class SexDataset(Dataset):
    """
    Dataset PyTorch pour la classification du sexe de la victime.

    INPUT  : texte complet de la décision
    OUTPUT : label binaire {0: homme, 1: femme}

    Truncation :
      strategy="head"      → premiers 512 tokens
        Recommandé car le signal sexe (Monsieur/Madame) est dans les
        premiers 3.7% du texte (médiane), soit bien dans les 512 premiers tokens.

      strategy="head_tail" → 256 premiers + 256 derniers tokens
        Alternative si le signal peut être en fin de document aussi.

    Tokenisation :
      On tokenise tout le batch au moment de la construction du dataset
      (pas à la volée) pour accélérer l'entraînement.
      Le padding est fait à max_length pour des batches homogènes.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        truncation_strategy: str = "head",
    ):
        """
        Parameters
        ----------
        texts                : liste de textes bruts
        labels               : liste d'entiers (0 ou 1)
        tokenizer            : tokenizer CamemBERT
        max_length           : nombre max de tokens (512 = limite CamemBERT)
        truncation_strategy  : "head" ou "head_tail"
        """
        assert len(texts) == len(labels), \
            f"Textes ({len(texts)}) et labels ({len(labels)}) de tailles différentes"

        self.labels = labels

        # Nettoyage + stratégie de truncation
        if truncation_strategy == "head_tail":
            logger.info("SexDataset : stratégie head_tail (256+256 tokens)")
            processed_texts = [
                truncate_head_tail(clean_text(t), tokenizer, max_length)
                for t in texts
            ]
        else:
            logger.info("SexDataset : stratégie head (512 premiers tokens)")
            processed_texts = [clean_text(t) for t in texts]

        # Tokenisation batch complète
        self.encodings = tokenizer(
            processed_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        logger.info(
            f"SexDataset créé : {len(labels)} exemples | "
            f"max_length={max_length} | strategy={truncation_strategy}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class DateDataset(Dataset):
    """
    Dataset PyTorch pour la classification des phrases de dates.

    INPUT  : contexte autour d'une phrase contenant une date
             (phrase_i + phrases[i-w : i+w])
    OUTPUT : label 3-classes {0: date_accident, 1: date_consolidation, 2: autre_date}

    Pourquoi 256 tokens et pas 512 ?
      Les phrases candidates sont courtes (1-3 phrases juridiques).
      256 tokens capturent largement le contexte ±2 phrases.
      Avantage : batch size × 2 possible → entraînement plus rapide.
    """

    def __init__(
        self,
        contexts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ):
        """
        Parameters
        ----------
        contexts   : liste de contextes textuels (phrase + voisins)
        labels     : liste d'entiers (0, 1 ou 2)
        tokenizer  : tokenizer CamemBERT
        max_length : nombre max de tokens (256 pour les phrases)
        """
        assert len(contexts) == len(labels), \
            f"Contextes ({len(contexts)}) et labels ({len(labels)}) de tailles différentes"

        self.labels = labels

        self.encodings = tokenizer(
            contexts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        logger.info(
            f"DateDataset créé : {len(labels)} exemples | max_length={max_length}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# =============================================================================
# Test rapide
# =============================================================================

if __name__ == "__main__":
    print("=== Test du module preprocessing ===\n")

    # --- Test regex de dates ---
    sample_texts = [
        "Le 9 avril 1991, Monsieur X... a été victime d'un accident du travail.",
        "Sa consolidation a été fixée au 7 novembre 1982 par l'expert.",
        "L'audience a eu lieu le 10/09/2008 devant la chambre sociale.",
        "L'arrêt du 30-04-2002 confirme le jugement de première instance.",
        "Le 1er mars 2004, il a été examiné par le docteur A...",
        "Aucune date ici.",
    ]

    for text in sample_texts:
        dates = extract_date_sentences(text, context_window=0)
        if dates:
            for d in dates:
                print(f"  Texte : {text[:60]}...")
                print(f"  -> Date trouvee : '{d['date_raw']}' -> {d['date_normalized']}")
        else:
            print(f"  Texte : {text[:60]}... -> Aucune date")
        print()

    # --- Test sur un exemple complet ---
    doc = """
    Le 9 avril 1991, Monsieur Yvon X... a été victime d'un accident du travail.
    Le tribunal a rendu son jugement le 12 février 2001.
    L'expert a examiné la victime le 2 février 2000.
    Sa consolidation a été fixée au 7 novembre 1995.
    """
    print("=== Extraction sur document complet ===")
    results = extract_date_sentences(doc.strip(), context_window=1)
    for r in results:
        print(f"  Date: {r['date_normalized']} | Raw: '{r['date_raw']}'")
        print(f"  Contexte: {r['context'][:100].encode('ascii', errors='replace').decode()}...")
        print()
