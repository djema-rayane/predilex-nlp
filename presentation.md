# Predilex NLP — Présentation complète
## Extraction automatique d'informations dans des décisions judiciaires françaises
### Projet M2 Deep NLP — Fine-tuning CamemBERT

---

## SLIDE 1 — Contexte : Le challenge Predilex

### Qu'est-ce que Predilex ?
Predilex est un projet de recherche en droit social qui vise à analyser automatiquement
la jurisprudence française des cours d'appel.

Les décisions de justice sont **publiques** mais **non structurées** :
- Des milliers de pages de texte brut
- Aucune base de données structurée
- Un juriste met 20 à 30 minutes pour lire et extraire les informations d'une seule décision

### Le problème concret
Pour analyser des tendances jurisprudentielles (évolution des indemnisations,
durées de consolidation par type d'accident...), il faudrait lire des milliers de décisions.
C'est humainement impossible à grande échelle.

### Notre objectif
Automatiser l'extraction de 3 informations clés dans chaque décision :
- Le **sexe** de la victime (homme / femme)
- La **date de l'accident** (YYYY-MM-DD ou "n.c." si non trouvée)
- La **date de consolidation** (YYYY-MM-DD ou "n.c." si non applicable)

---

## SLIDE 2 — Comprendre les données

### De quels documents s'agit-il ?
Des décisions de **cours d'appel françaises** portant sur des **accidents du travail**.

Un salarié se blesse au travail. Il attaque son employeur ou la sécurité sociale
pour obtenir réparation. Le juge rend une décision écrite.

**Exemple réel — Agen_100515.txt :**
> "Le 9 avril 1991, Monsieur Yvon X... a été victime d'un accident du travail
> au cours duquel il a été gravement blessé."

Ce document fait partie d'un litige entre Monsieur X (victime), la société
de transport MOREL (son employeur) et Azur Assurances. Le tribunal a reconnu
la **faute inexcusable** de l'employeur.

### Pourquoi la consolidation est-elle importante ?
La date de consolidation est le moment où les blessures sont médicalement
considérées comme "stabilisées" — plus d'évolution possible.

- **Avant la consolidation** → indemnités journalières (arrêt de travail)
- **Après la consolidation** → rente à vie si séquelles permanentes (IPP)

Plus la consolidation est tardive, plus l'indemnisation est élevée.
C'est donc une information juridiquement et financièrement cruciale.

### Statistiques clés du dataset

| Caractéristique | Valeur |
|---|---|
| Nombre de documents | 770 décisions |
| Longueur moyenne | 3 315 mots par document |
| Longueur maximale | > 10 000 mots |
| Tous > 512 tokens | OUI — contrainte technique forte |
| Juridictions | Agen, Aix-en-Provence, Paris, Versailles, Toulouse... |
| Période | Années 1980 à 2020 |

### Distribution des labels

**Sexe :**
- 559 hommes (73%)
- 206 femmes (27%)
- 5 non renseignés (n.c.) → exclus de l'évaluation

**Date d'accident :**
- ~747 documents avec une vraie date (97%)
- ~23 documents "n.c." (3%)
- Quasi toujours présente : c'est le fait central du litige

**Date de consolidation :**
- ~447 documents avec une vraie date (58%)
- ~323 documents "n.c." (42%)
- Absente si accident récent, ou si le juge n'a pas eu à se prononcer

---

## SLIDE 3 — Analyse exploratoire des documents

### Ce qu'on a découvert en lisant les documents

**1. Le signal sexe apparaît très tôt dans le texte**
- "Monsieur / Madame" apparaît en médiane à **3.7% du texte**
- Les 512 premiers tokens capturent ce signal dans **94% des cas**
- Conclusion : on n'a pas besoin de lire tout le document pour prédire le sexe

**2. Il y a en moyenne 33 dates par document**
- Dates de jugement, dates de rapports médicaux, dates d'expertises...
- On ne peut pas donner les 3300 mots au modèle (limite 512 tokens)
- Il faut **réduire le problème** : extraire seulement les phrases avec des dates

**3. Les formats de dates sont très variés**
- Textuel : "9 avril 1991", "1er mars 2004", "neuf avril mil neuf cent..."
- Numérique : "09/04/1991", "09-04-1991", "02.05.2000"
- Format liste : "- consolidation : 02.05.2000"
- Tronqué : "date de consolidation le" (suite sur la ligne suivante)

**4. Patterns récurrents pour la consolidation**
- Pattern A (40%) : "- date de consolidation : 12 septembre 2003" → court, direct
- Pattern B (30%) : "La date de consolidation a été fixée au 8 avril 1999" → phrase complète
- Pattern C (20%) : date implicite, "consolidation" dans la phrase précédente
- Pattern D (10%) : phrase tronquée, date sur la ligne d'après → très difficile

**5. "Consolidation" précède presque toujours la date**
- Le mot-clé vient avant la date, rarement après
- Contexte asymétrique : regarder plus loin avant qu'après

---

## SLIDE 4 — Choix du modèle : pourquoi CamemBERT ?

### Qu'est-ce que CamemBERT ?
CamemBERT est un modèle de langue pré-entraîné **exclusivement sur du texte français**.

- Architecture : **RoBERTa** (version améliorée de BERT, 2019)
- Développé par : **INRIA + Facebook AI Research**
- Pré-entraîné sur : **138 GB de texte français**
  (Common Crawl, Wikipedia FR, livres numériques, presse...)
- Nombre de paramètres : **110 millions**
- Tokenisation : **SentencePiece** (meilleure gestion des mots composés français)

### Pourquoi CamemBERT et pas un autre modèle ?

**vs XLM-RoBERTa (multilingue) :**
- XLM-RoBERTa gère 100 langues → son vocabulaire est partagé entre 100 langues
- CamemBERT consacre 100% de son vocabulaire au français
- Meilleure tokenisation du vocabulaire juridique français
- "inexcusable", "consolidation", "indemnisation" → mieux représentés

**vs CamemBERT-large (330M params) :**
- 770 documents = dataset PETIT pour le Deep Learning
- Un modèle 3x plus grand → overfitting quasi certain
- La vraie différence viendra du pipeline, pas de la taille

**vs entraîner un modèle from scratch :**
- Impossible avec 770 documents — il en faudrait des millions
- Le transfer learning résout ce problème

### Le principe du Transfer Learning

CamemBERT a déjà "lu" 138 GB de texte français.
Il comprend la syntaxe, la sémantique, les nuances du français.

Le fine-tuning consiste à lui montrer nos 770 documents pour qu'il apprenne
la sémantique spécifique des accidents du travail — sans repartir de zéro.

Analogie : former un juriste expert en français à lire des contrats d'assurance.
Il n'a pas besoin de réapprendre à lire — juste à reconnaître les patterns spécifiques.

---

## SLIDE 5 — Architecture globale : pipeline hybride

### Le défi principal
Tous nos documents font plus de 512 tokens (limite dure des transformers).
CamemBERT ne peut pas lire un document entier.

### La solution : pipeline hybride NLP classique + Deep Learning

```
Document entier (3300 mots)
         |
         |------------------------------------------|
         |                                          |
         v                                          v
[MODÈLE 1 — SEXE]                     [MODÈLE 2 — DATES]
512 premiers tokens                   Etape 1 : NLP Classique
CamemBERT fine-tuné                   Regex → 33 phrases candidates
2 classes : homme / femme             Normalisation YYYY-MM-DD
WeightedLoss (73/27)                          |
                                              v
                                    Etape 2 : Deep Learning
                                    CamemBERT fine-tuné
                                    Input : phrase + contexte +-2
                                    3 classes :
                                    - date_accident
                                    - date_consolidation
                                    - autre_date
                                              |
                                              v
                                    Etape 3 : Logique métier
                                    Score max par catégorie
                                    + Seuil de confiance
                                    → date ou n.c.
```

### Pourquoi ce pipeline est pertinent ?

**NLP classique** (Regex) :
- Déterministe, rapide, couvre tous les formats de dates
- Réduit 3300 mots → 33 phrases candidates
- Normalise les dates vers YYYY-MM-DD

**Deep Learning** (CamemBERT) :
- Comprend le sens sémantique de chaque phrase
- Distingue "victime d'un accident le 9 avril" de "jugement rendu le 9 avril"
- Gère les formulations implicites et le contexte

**Les deux sont nécessaires :**
- Le Regex seul ne comprend pas le sens → confond les dates
- CamemBERT seul ne peut pas lire 3300 mots → limite 512 tokens
- Ensemble : le Regex fait la réduction, CamemBERT fait la compréhension

---

## SLIDE 6 — Modèle 1 : Classification du sexe

### Input / Output
- **Input** : les 512 premiers tokens du document
- **Output** : homme (0) ou femme (1)

### Choix techniques justifiés

**Truncation "head" (512 premiers tokens) :**
Signal "Monsieur/Madame" apparaît à 3.7% du texte en médiane.
Les 512 premiers tokens capturent ce signal dans 94% des cas.
Pas besoin de lire tout le document.

**WeightedTrainer — CrossEntropyLoss pondérée :**
Problème : 73% d'hommes → un modèle non pondéré apprend à tout prédire "homme".
Solution : on pondère inversement à la fréquence.
- Poids homme = 0.69 (moins pénalisé)
- Poids femme = 1.88 (plus pénalisé)

```python
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = CrossEntropyLoss(weight=self.class_weights.to(device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
```

**Early stopping sur F1 Weighted (patience=2) :**
Arrêt automatique si le F1 ne s'améliore pas → évite l'overfitting.

### Résultats

| | Baseline "toujours homme" | Notre modèle CamemBERT |
|---|---|---|
| Accuracy | 73.33% | **90.67%** |
| F1 Macro | 0.42 | **0.89** |
| Recall femme | 0% | **95%** |
| Précision femme | - | 76% |

**Gain : +17 points** grâce au fine-tuning.

Le modèle a appris à reconnaître "Monsieur/Madame", "il/elle",
les prénoms féminins/masculins — et non pas à mémoriser la distribution 73/27.

**Matrice de confusion :**
- 49 hommes correctement classés
- 19 femmes correctement classées
- 6 hommes classés femme (faux positifs)
- 1 femme classée homme (faux négatif)

---

## SLIDE 7 — Modèle 2 : Extraction des dates — Détail du pipeline

### Etape 1 : NLP Classique — Extraction par Regex

Expression régulière couvrant tous les formats de dates françaises rencontrés :
- Textuel : "14 juin 1999", "1er mars 2004", "quatorze juin..."
- Slash : "14/06/1999", "14/6/99"
- Tiret : "14-06-1999"
- Point : "14.06.1999"

Pour chaque date trouvée :
1. On identifie la phrase qui la contient
2. On prend les 2 phrases avant et après (fenêtre de contexte ±2)
3. On normalise la date vers YYYY-MM-DD

**Résultat : ~33 phrases candidates par document en moyenne**

### Etape 2 : Deep Learning — Classification CamemBERT

**Input :** [phrase_i-2] + [phrase_i-1] + [phrase_avec_date] + [phrase_i+1] + [phrase_i+2]
Tokenisé en **max 256 tokens** (phrases courtes → on peut utiliser des batches 2x plus grands)

**Pourquoi le contexte est crucial :**
> "L'expert a conclu à une consolidation."  ← phrase i-1
> "La date retenue est le 7 novembre 1998." ← phrase i (contient la date)

Sans la phrase i-1, CamemBERT ne sait pas que cette date est une consolidation.

**3 classes à prédire :**
- `date_accident` (0) : "le 9 avril 1991, M. X a été victime d'un accident..."
- `date_consolidation` (1) : "consolidation fixée au 7 novembre 1998..."
- `autre_date` (2) : date de jugement, rapport médical, expertise...

**Déséquilibre extrême des classes :**
- date_accident : 3% des phrases
- date_consolidation : 2% des phrases
- autre_date : **95% des phrases**

Sans correction → le modèle prédit tout "autre_date" (95% d'accuracy, 0% d'utilité).

**Solution : Oversampling des classes rares (ratio 0.3)**
- On duplique les exemples accident et consolidation
- Jusqu'à atteindre 30% de la classe majoritaire
- Le modèle voit suffisamment d'exemples positifs pour apprendre

**Métrique d'optimisation : F1 Macro**
F1 Macro pénalise également les erreurs sur toutes les classes,
même les rares (accident, consolidation).
Si on utilisait l'accuracy, 95% serait atteignable sans rien apprendre.

### Etape 3 : Logique métier — Seuils de décision

Pour chaque document :
1. On récupère les scores de toutes les phrases candidates
2. On sélectionne la phrase avec le **meilleur score** pour chaque catégorie
3. On applique un seuil de confiance :

**threshold_accident = 0.55 :**
Si le meilleur score accident < 0.55 → prédire "n.c."
(le modèle n'est pas assez confiant → mieux vaut ne pas répondre)

**threshold_consolidation = 0.45 :**
Plus bas car 42% des docs ont une vraie consolidation →
on peut se permettre d'être moins exigeant.

---

## SLIDE 8 — Résultats

| | Baseline NLP classique (regex + mots-clés) | **Notre modèle (NLP + CamemBERT)** | Gain |
|---|---|---|---|
| Sexe | 73.33% (toujours homme) | **90.67%** | **+17 pts** |
| date_accident | 30.67% | **77.92%** | **+47 pts** |
| date_consolidation | 69.33% | **74.03%** | **+5 pts** |

### Pourquoi ces gains ?

**Sexe (+17 pts) :**
Le modèle a appris les signaux textuels (Monsieur/Madame, prénoms, pronoms).
La WeightedLoss lui a permis de ne pas ignorer la classe minoritaire (femme).

**Accident (+47 pts) :**
C'est le gain le plus impressionnant. Le mot "accident" apparaît des dizaines
de fois dans chaque document. La baseline prend la première occurrence → souvent
une mauvaise date. CamemBERT comprend quelle phrase **définit** la date d'accident
vs celles qui y font simplement **référence**.

**Consolidation (+5 pts) :**
Gain plus modeste car la baseline mots-clés est déjà forte ici.
Le mot "consolidation" est très spécifique — quand il apparaît avec une date,
c'est presque toujours la bonne. CamemBERT améliore surtout les cas implicites
et la prédiction n.c. (97% vs 88% pour la baseline).

---

## SLIDE 9 — Analyse des erreurs

### Pourquoi le modèle se trompe sur l'accident ?

**Cas 1 — Confusion avec date de jugement :**
> "Par jugement du 24 avril 1997, le tribunal a reconnu..."
> "Suite à l'accident du 24 avril 1997..."  ← même date, contexte différent

CamemBERT confond parfois ces deux types de phrases.

**Cas 2 — Date formulée sans contexte :**
> "Victime d'un accident survenu en 1991" ← sans le jour exact
Le regex extrait une date approximative qui ne correspond pas au CSV.

### Pourquoi le modèle se trompe sur la consolidation ?

**Cas 1 — Formulation implicite :**
> "Le 15 mars 2010, le docteur A... a examiné Monsieur X..."
La consolidation est implicite (date de l'examen final) — pas le mot "consolidation".

**Cas 2 — Phrase tronquée :**
> "- IPP 1% - date de consolidation le"
La date est sur la ligne suivante → le contexte ±2 ne la capture pas toujours.

**Cas 3 — Plusieurs dates de consolidation mentionnées :**
Certains documents discutent de consolidations successives ou contestées.
Le modèle peut prendre la mauvaise.

### Matrice de confusion — Niveau phrase

|  | Prédit accident | Prédit consolidation | Prédit autre |
|---|---|---|---|
| Réel accident | 135 | 17 | 36 |
| Réel consolidation | 1 | 51 | 22 |
| Réel autre | 143 | 62 | 2029 |

143 phrases "autre_date" classées comme accident → faux positifs.
La logique d'agrégation (prendre le max sur tout le document) compense
une partie de ces erreurs au niveau document.

---

## SLIDE 10 — Réponses aux questions du jury

### "770 documents c'est peu pour du Deep Learning, pourquoi ce choix ?"

**Argument 1 — Transfer Learning :**
CamemBERT a été pré-entraîné sur 138 GB de français.
Les 770 documents ne servent pas à lui apprendre le français —
ils servent à lui apprendre la sémantique spécifique des accidents du travail.

**Argument 2 — Le NLP classique ne suffit pas :**
La baseline 3 (NLP seul) fait **30.67%** sur l'accident.
Notre modèle fait **77.92%**. Les +47 points justifient le Deep Learning.

**Argument 3 — 770 docs × 33 dates = 25 000 phrases candidates**
Le classifieur de dates est entraîné sur ~25 000 phrases, pas 770 documents.
C'est une taille raisonnable pour du fine-tuning.

---

### "Pourquoi ne pas avoir utilisé un LLM (GPT, Mistral, LLaMA) ?"

- Les LLMs nécessitent des API payantes ou des GPU très puissants (80 GB VRAM)
- Le fine-tuning de CamemBERT tourne sur un GPU T4 gratuit (Colab/Kaggle)
- Pour une tâche de classification structurée, un modèle fine-tuné
  bat souvent un LLM en zero-shot sur des données spécialisées
- CamemBERT est le standard pour le NLP juridique français dans la littérature

---

### "La précision phrase-level est faible (53% accident), c'est normal ?"

Oui, pour deux raisons :
1. Le déséquilibre 1:25 — même avec oversampling, la tâche reste difficile
2. Ce n'est pas la métrique qui compte — ce qui compte c'est l'accuracy document

La logique d'agrégation compense : même si le modèle fait des erreurs
sur certaines phrases, il classe généralement la vraie phrase d'accident
plus haut que les fausses → 78% au niveau document.

---

### "Comment améliorer les résultats ?"

1. **Plus de données** — le levier principal. Avec 5000 documents, les résultats
   seraient significativement meilleurs.
2. **Fenêtre de contexte asymétrique** — before=3, after=1 pour la consolidation
   (le mot-clé précède toujours la date)
3. **Modèle juridique spécialisé** — CamemBERT pré-entraîné sur des textes
   juridiques français (en cours de développement dans la communauté NLP)
4. **Meilleur découpage des phrases** — fusionner les phrases tronquées
   avant extraction

---

## SLIDE 11 — Conclusion

### Ce qu'on a réalisé
- Un pipeline complet de A à Z : de la donnée brute à la prédiction finale
- 2 modèles CamemBERT fine-tunés sur du vocabulaire juridique français
- Architecture hybride NLP classique + Deep Learning justifiée et validée
- Analyse rigoureuse des erreurs et itérations sur les hyperparamètres

### Résultats finaux

| Tâche | Baseline | Notre modèle | Gain |
|---|---|---|---|
| Sexe | 73.33% | **90.67%** | +17 pts |
| date_accident | 30.67% | **77.92%** | +47 pts |
| date_consolidation | 69.33% | **74.03%** | +5 pts |

### Points forts du projet
- Le transfer learning permet de faire du Deep Learning efficace sur 770 documents
- Le pipeline hybride contourne la limite des 512 tokens des transformers
- La calibration des seuils permet de prédire "n.c." avec 97-100% de précision

### Limites honnêtes
- La précision phrase-level reste faible (53% accident) — tâche difficile
- Les formulations implicites de consolidation restent un défi
- Plus de données améliorerait significativement tous les scores
