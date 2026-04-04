"""
model_utils.py
--------------
Utilitaires pour le modèle CamemBERT : chargement, sauvegarde, inférence.

Ce module centralise tout ce qui touche au modèle :
  - Détection automatique du device (GPU CUDA > CPU)
  - Chargement du tokenizer et du modèle CamemBERT
  - Sauvegarde HuggingFace (format compatible from_pretrained)
  - Inférence par batch (prédictions + probabilités)
  - Comptage des paramètres entraînables

Architecture CamemBERT :
  CamemBERT = RoBERTa adapté au français
    → Encodeur transformer 12 couches, 768 dimensions cachées
    → Tokenisation SentencePiece (vocabulaire 32 000 tokens)
    → Pré-entraîné sur 138 GB de texte français (Oscar + Common Crawl FR)
    → 110M paramètres au total, ~85M entraînables pour fine-tuning

Pour la classification :
  CamembertForSequenceClassification ajoute une tête de classification
  (Linear + dropout) sur le token [CLS] de la dernière couche.
  C'est cette représentation du [CLS] qui encode le sens global de la séquence.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    CamembertForSequenceClassification,
    CamembertTokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Détection du device
# =============================================================================

def get_device() -> torch.device:
    """
    Retourne le meilleur device disponible.

    Ordre de priorité :
      1. CUDA (GPU NVIDIA) : entraînement ~10-30x plus rapide
      2. CPU                : fallback universel

    Note : MPS (Apple Silicon) retiré car instable avec certaines ops
    de CamemBERT sur PyTorch < 2.1.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU détecté : {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        logger.info(
            "Aucun GPU CUDA détecté — utilisation du CPU.\n"
            "  → L'entraînement sera LENT (30-60 min par époque sur CPU).\n"
            "  → Utilisez Google Colab ou un serveur GPU si possible."
        )
    return device


def is_fp16_available() -> bool:
    """
    Vérifie si l'entraînement en mixed precision (fp16) est possible.

    fp16 (half precision) divise par ~2 la mémoire GPU utilisée
    et accélère l'entraînement de ~30-50% sur GPU NVIDIA avec Tensor Cores.
    Disponible uniquement sur GPU CUDA (pas sur CPU).
    """
    return torch.cuda.is_available()


# =============================================================================
# Chargement du tokenizer
# =============================================================================

def load_tokenizer(model_name: str = "camembert-base") -> CamembertTokenizerFast:
    """
    Charge le tokenizer CamemBERT.

    Le tokenizer SentencePiece de CamemBERT :
      - Vocabulaire de 32 000 sous-mots (subwords)
      - Gère les mots inconnus par décomposition (pas d'UNK unique)
      - Sensible à la casse et aux accents français
      - Tokens spéciaux : <s> (CLS), </s> (SEP), <pad>, <unk>, <mask>

    Parameters
    ----------
    model_name : identifiant HuggingFace ou chemin local
                 Default: "camembert-base" (télécharge depuis HF Hub)
    """
    logger.info(f"Chargement tokenizer : {model_name}")
    tokenizer = CamembertTokenizerFast.from_pretrained(
        model_name,
        use_fast=True,  # version rapide (Rust) vs Python
    )
    logger.info(
        f"Tokenizer prêt | vocab_size={tokenizer.vocab_size:,} | "
        f"max_model_length={tokenizer.model_max_length}"
    )
    return tokenizer


# =============================================================================
# Chargement du modèle
# =============================================================================

def load_model(
    model_name: str = "camembert-base",
    num_labels: int = 2,
    label2id: Optional[Dict[str, int]] = None,
    id2label: Optional[Dict[int, str]] = None,
) -> CamembertForSequenceClassification:
    """
    Charge CamemBERT pré-entraîné et ajoute une tête de classification.

    Architecture résultante :
      [CamemBERT encoder] → [pooler sur token CLS] → [dropout] → [Linear(768, num_labels)]

    Le poids de la tête de classification est initialisé aléatoirement.
    Les poids de l'encodeur sont chargés depuis le checkpoint pré-entraîné.
    Le fine-tuning va adapter TOUS les poids (encodeur + tête) à notre tâche.

    Parameters
    ----------
    model_name  : identifiant HuggingFace ou chemin local
    num_labels  : nombre de classes (2 pour sexe, 3 pour dates)
    label2id    : mapping label_name → int  (stocké dans la config du modèle)
    id2label    : mapping int → label_name  (utilisé pour l'affichage des prédictions)

    ignore_mismatched_sizes=True : nécessaire car la tête de classification
    du checkpoint pré-entraîné (MLM) a une taille différente de la nôtre.
    """
    logger.info(f"Chargement modèle : {model_name} | {num_labels} classes")

    model = CamembertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    return model


def load_trained_model(
    checkpoint_path: str,
    num_labels: int = 2,
) -> CamembertForSequenceClassification:
    """
    Charge un modèle déjà fine-tuné depuis un checkpoint local.

    Utilisé dans predict.py pour l'inférence finale sur le test set.

    Parameters
    ----------
    checkpoint_path : chemin vers le dossier contenant config.json + pytorch_model.bin
    num_labels      : doit correspondre au modèle sauvegardé
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {checkpoint_path}\n"
            "Lancez d'abord l'entraînement : python src/train_sex.py"
        )
    logger.info(f"Chargement modèle fine-tuné : {checkpoint_path}")
    model = CamembertForSequenceClassification.from_pretrained(
        checkpoint_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=False,  # le checkpoint a déjà la bonne taille
    )
    return model


# =============================================================================
# Sauvegarde
# =============================================================================

def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str,
) -> None:
    """
    Sauvegarde le modèle et le tokenizer au format HuggingFace.

    Le format HuggingFace permet de recharger le modèle avec :
        model = CamembertForSequenceClassification.from_pretrained(output_dir)
        tokenizer = CamembertTokenizerFast.from_pretrained(output_dir)

    Fichiers créés :
        config.json          : architecture + hyperparamètres
        pytorch_model.bin    : poids du modèle (ou model.safetensors)
        tokenizer_config.json
        sentencepiece.bpe.model : vocabulaire SentencePiece
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Modèle sauvegardé dans : {output_dir}")


# =============================================================================
# Introspection du modèle
# =============================================================================

def count_parameters(model: PreTrainedModel) -> int:
    """
    Compte et affiche les paramètres du modèle.

    Utile pour vérifier que le modèle est chargé correctement et
    comprendre la taille de ce qu'on entraîne.

    Pour camembert-base :
      - Total : ~111M paramètres
      - Entraînables : ~111M (tous fine-tunés, sauf si on freeze des couches)

    Returns
    -------
    Nombre de paramètres entraînables
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    logger.info(
        f"Paramètres du modèle :\n"
        f"  Total      : {total:>12,}\n"
        f"  Entraînable: {trainable:>12,}\n"
        f"  Gelés      : {frozen:>12,}"
    )
    return trainable


# =============================================================================
# Inférence par batch
# =============================================================================

def predict_batch(
    texts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 16,
) -> List[int]:
    """
    Prédictions de classe pour une liste de textes.

    Traitement par batch pour éviter les OOM sur GPU.

    Returns
    -------
    Liste d'entiers (classe prédite pour chaque texte)
    """
    preds, _ = predict_batch_with_probs(
        texts, model, tokenizer, device, max_length, batch_size
    )
    return preds


def predict_batch_with_probs(
    texts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    max_length: int = 512,
    batch_size: int = 16,
) -> Tuple[List[int], np.ndarray]:
    """
    Prédictions + probabilités (softmax) pour une liste de textes.

    Les probabilités sont nécessaires pour :
      - Calibrer le seuil de consolidation dans train_dates.py
      - Afficher la confiance du modèle dans predict.py
      - Choisir la meilleure phrase parmi les candidats

    Parameters
    ----------
    texts      : liste de textes à classer
    model      : modèle CamemBERT fine-tuné
    tokenizer  : tokenizer correspondant
    device     : cuda ou cpu
    max_length : longueur max en tokens
    batch_size : taille des batches pour l'inférence

    Returns
    -------
    preds  : List[int]      — classe prédite pour chaque texte
    probs  : np.ndarray     — shape (n_texts, n_classes), probabilités softmax
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # (batch, n_classes)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1).tolist()

        all_preds.extend(preds)
        all_probs.append(probs)

    all_probs_arr = np.vstack(all_probs) if all_probs else np.array([])
    return all_preds, all_probs_arr


# =============================================================================
# Test rapide
# =============================================================================

if __name__ == "__main__":
    print("=== Test model_utils ===\n")

    device = get_device()
    print(f"Device sélectionné : {device}\n")

    print("Chargement tokenizer...")
    tokenizer = load_tokenizer("camembert-base")

    print("\nChargement modèle (2 classes)...")
    model = load_model(
        model_name="camembert-base",
        num_labels=2,
        label2id={"homme": 0, "femme": 1},
        id2label={0: "homme", 1: "femme"},
    )
    count_parameters(model)

    print("\nTest d'inférence...")
    test_texts = [
        "Monsieur Yvon X... a été victime d'un accident du travail.",
        "Madame Marie X... a subi un accident de la route le 14 juin 1999.",
    ]
    preds, probs = predict_batch_with_probs(
        test_texts, model, tokenizer, device, max_length=64, batch_size=2
    )
    for text, pred, prob in zip(test_texts, preds, probs):
        label = "homme" if pred == 0 else "femme"
        print(f"  '{text[:50]}...'")
        print(f"  → Prédiction : {label} | Confiance : {prob[pred]:.3f}")
        print()
