#  Détection de discopathie dégénérative par segmentation et classification d'images IRM lombaires

##  Objectif
Ce projet propose un pipeline complet d'aide au diagnostic de la discopathie dégénérative à partir d'IRM lombaires (séquences T2), en combinant :
1. la **segmentation** automatique des disques intervertébraux,
2. la **classification** binaire de chaque disque (sain vs. pathologique) à partir du grade de Pfirrmann.

##  Problématique
La discopathie dégénérative est une pathologie fréquente dont le diagnostic repose sur l'analyse visuelle d'IRM par un radiologue. Ce projet explore comment un pipeline d'IA (segmentation + classification) peut fournir une aide au diagnostic automatisée et reproductible.

##  Méthodologie

### 1. Préparation des données
- Lecture des volumes IRM et masques au format `.mha` via **SimpleITK**
- Filtrage des séquences **T2** et des disques **lombaires** (niveaux IVD 1 à 5)
- Construction d'un label binaire à partir du **grade de Pfirrmann** : `0` = sain (grade I-II), `1` = discopathie dégénérative (grade III-V)
- Split **train/validation stratifié par patient** (80/20, `GroupShuffleSplit`) pour éviter toute fuite de données entre les deux ensembles
- Contrôle qualité du signal : calcul du **SNR** (rapport signal/bruit) sur les volumes IRM pour valider la qualité des données en entrée

### 2. Segmentation (U-Net 2D)
- Architecture **U-Net 2D** codée from scratch (encodeur-décodeur avec skip connections, 4 niveaux, base=32 canaux)
- Entrée : la **slice la plus informative** de chaque volume (celle contenant la plus grande surface de disque), normalisée (z-score) et recadrée en 256×256
- Fonction de perte : **BCE + Dice Loss** combinées
- Entraînement : 20 epochs, optimiseur AdamW (lr=1e-3)

### 3. Classification (ResNet18 2.5D)
- Extraction d'une **ROI 2.5D** autour de chaque disque : 5 slices adjacentes empilées comme canaux (128×128, marge de 12 px autour du disque segmenté)
- Modèle **ResNet18** adapté pour une entrée à 5 canaux (première couche convolutionnelle modifiée), sortie binaire
- Gestion du déséquilibre de classes via **WeightedRandomSampler**
- Entraînement : 15 epochs, perte BCEWithLogits
- Recherche du **seuil de décision optimal** par maximisation de la balanced accuracy
- Agrégation finale des prédictions au **niveau patient** (probabilité maximale parmi ses disques)

##  Dataset
- **447 volumes IRM** (séries T1/T2) et **450 masques** de segmentation (format `.mha`)
- Après filtrage T2 + disques lombaires : **1060 disques annotés** (grade de Pfirrmann + labels radiologiques : Modic, pincement discal, hernie, bombement, spondylolisthésis)
- Split : **845 disques / 169 patients** en train, **215 disques / 43 patients** en validation
- Distribution des classes (val) : 146 disques pathologiques vs 69 sains — déséquilibre géré par le sampler

Dataset **SPIDER** (Segmentation of the sPIne for Degenerative conditions in Radiology) : IRM lombaires multi-centres avec segmentations de référence des vertèbres, disques intervertébraux et canal spinal.
- Source : [SPIDER Grand Challenge](https://spider.grand-challenge.org/) · [Zenodo](https://zenodo.org/records/8009680)
- Accès utilisé : [Kaggle mirror](https://www.kaggle.com/datasets/dankok/spider-lumbar-spine-segmentation-in-mr-images)

##  Licence des données
Le dataset SPIDER est distribué sous licence **CC BY 4.0** (Creative Commons Attribution 4.0). Utilisation libre sous réserve de citer les auteurs originaux :

> van der Graaf, J.W., van Hooff, M.L., Buckens, C.F.M. et al. (2023). *Lumbar spine segmentation in MR images: a dataset and a public benchmark.* arXiv:2306.12217

##  Résultats

**Segmentation (U-Net 2D)**
| Métrique | Valeur |
|---|---|
| Dice score (meilleur modèle, validation) | **0.857** |
| Dice score (exemple image spécifique) | 0.917 |

**Classification — niveau disque (slice-level)**
| Métrique | Valeur |
|---|---|
| AUC | **0.802** |
| Accuracy | 0.786 |
| F1-score | 0.740–0.848 selon seuil |
| Balanced Accuracy | 0.746 (seuil optimisé = 0.95) |

**Classification — niveau patient (agrégation finale)**
| Métrique | Valeur |
|---|---|
| Balanced Accuracy | **0.951** |
| F1-score | **0.949** |

> L'agrégation au niveau patient améliore nettement la performance par rapport au niveau disque isolé, ce qui est cohérent cliniquement : le diagnostic final porte sur le patient, pas sur une seule coupe.

## Stack technique
`Python` · `PyTorch` / `torchvision` (ResNet18) · `SimpleITK` (imagerie médicale .mha) · `scikit-learn` (split, métriques) · `pandas` / `numpy` · `matplotlib` (courbes ROC, matrices de confusion) · Google Colab (GPU)

## Structure du repo
├── notebook/ # notebook Colab complet (segmentation + classification)
├── results/ # courbes ROC, matrices de confusion, métriques
└── README.md

##  Pistes d'amélioration
- Segmentation 3D (U-Net 3D) plutôt que 2D sur slice unique
- Validation croisée patient-level pour des métriques plus robustes
- Interprétabilité (Grad-CAM) pour visualiser les zones influençant la classification
- Test sur un jeu de données externe pour évaluer la généralisation

##  Auteure
**Hajar LKAD** — Étudiante en Génie Digital & Intelligence Artificielle appliquée à la santé, SupTech Santé
