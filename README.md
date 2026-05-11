# Analyse automatisée de lysosomes en microscopie de fluorescence

Pipeline complet d'apprentissage automatique pour la **détection, la classification
et le suivi temporel de lysosomes** dans des images de microscopie de fluorescence.

---

## Vue d'ensemble du projet

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Include.ipynb                                  │
│          (bibliothèques, classes, fonctions utilitaires)                │
└────────────────────────────┬────────────────────────────────────────────┘
                             │  %run "../Include.ipynb"
           ┌─────────────────┼──────────────────────┐
           │                 │                      │
           ▼                 ▼                      ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│ Pipeline_ML      │ │  Fluo_Analyse    │ │   Tracking           │
│ .ipynb           │ │  .ipynb          │ │   .ipynb             │
│                  │ │                  │ │                      │
│  ENTRAÎNEMENT    │ │  ANALYSE BATCH   │ │  SUIVI TEMPOREL      │
│  du CNN soma     │ │  (images fixes)  │ │  (stack multi-frames)│
└────────┬─────────┘ └──────────────────┘ └──────────────────────┘
         │
         │  soma_cnn_test.pth
         └────────────────────────►  Fluo_Analyse.ipynb
                                  └►  Tracking.ipynb
```

**Le modèle CNN (`soma_cnn_test.pth`) produit par `Pipeline_ML.ipynb`
est une dépendance obligatoire pour les deux autres notebooks.**

---

## Description des fichiers

| Fichier | Rôle |
|---|---|
| `Include.ipynb` | Socle commun : imports, classes (`ConvAutoEncoder`, `SomaCNN`, `SomaBinaryDataset`), et toutes les fonctions du pipeline |
| `Pipeline_ML.ipynb` | Entraînement du CNN soma (auto-encodeur → clustering → sélection manuelle → CNN supervisé) |
| `Fluo_Analyse.ipynb` | Analyse en production sur images fixes : comptage de lysosomes par soma + export CSV |
| `Tracking.ipynb` | Suivi temporel de lysosomes sur stack TIFF multi-frames + génération de 7 figures d'analyse |

---

## Prérequis

### Dépendances Python

```bash
%pip install scikit-image
%pip install torch torchvision torchaudio
%pip install cellpose==2.1.0
%pip install scikit-image==0.25.2
%pip install tifffile==2023.4.12
%pip install scipy==1.17.1
%pip install numpy==1.26.4
%pip install pandas==2.2.2
%pip install trackpy
%pip install tqdm
%pip install matplotlib
```

> Un **GPU CUDA** est fortement recommandé pour Cellpose et l'inférence CNN
> (×5 à ×20 plus rapide qu'en CPU). Le pipeline fonctionne néanmoins sur CPU.

### Ordre d'exécution

```
1. Pipeline_ML.ipynb   →  produit soma_cnn_test.pth
2. Fluo_Analyse.ipynb             →  analyse d'un lot d'images fixes
   et / ou
   Tracking.ipynb          →  suivi temporel sur stack TIFF
```

> `Fluo_Analyse.ipynb` et `Tracking.ipynb` sont **indépendants l'un de l'autre**
> et peuvent être exécutés dans n'importe quel ordre une fois le CNN entraîné.

---

## `Include.ipynb` — Socle commun

Chargé par tous les notebooks via `%run "Include.ipynb"`. Contient :

- **Imports** : PyTorch, NumPy, scikit-image, Cellpose, TrackPy, UMAP, etc.
- **Classes PyTorch** : `ConvAutoEncoder`, `SomaCNN`, `SomaBinaryDataset`, `PatchDataset`
- **Fonctions de construction du dataset** : `build_dataset_specific_from_folder`, `build_blob_dataset`
- **Fonctions de détection** : `Best_plan_determination`, `CNN_Patches_Construction`, `detect_candidates`
- **Fonctions de clustering / fusion** : `Clusterization_detected_patches`, `Finale_Fusion_patches`
- **Fonctions d'analyse** : `Extract_fluo_informations`, `Extract_fluo_informations_without_nucleus`
- **Fonctions Cellpose** : `Cellpose_Analyse_Count`, `Cellpose_Analyse_Track_GPU`
- **Fonctions TrackPy** : `TrackPy_Construction`, `TrackPy_Construction_CNN_Cellpose`, `Tracking_Lyso_TrackPy`
- **Fonctions de sauvegarde** : `Save_images_and_overlay`, `Save_final_patches`, `Creation_CSV`
- **Utilitaires** : `get_device`, `train_one_epoch`

---

## `Pipeline_ML.ipynb` — Entraînement du CNN

Pipeline non supervisé puis supervisé pour produire le modèle de détection de somas.

```
Images TIFF → Patches 64×64 → Auto-encodeur → UMAP + KMeans
                                                     │
                                        Inspection visuelle des clusters
                                                     │
                                       PatchSelector (validation manuelle)
                                                     │
                                        Entraînement SomaCNN (supervisé)
                                                     │
                                      Évaluation Precision-Recall + seuil
                                                     │
                                           soma_cnn_test.pth
```

**Sorties :**

| Fichier | Description |
|---|---|
| `soma_cnn_test.pth` | Poids du CNN entraîné |
| `patches_*.npy` | Dataset de patches extrait des TIFFs |
| `manual_soma_selection.npy` | Indices des patches validés manuellement |
| `manual_soma_patches.npy` | Patches positifs validés |

---

## `Fluo_Analyse.ipynb` — Analyse d'images fixes

Traite un lot de fichiers `.tif` et produit un CSV de résultats.

```
Fichier TIFF → Plan net → Stats brutes → Stats hors-soma
                               │
                    CNN (somas) → Cellpose (lysosomes)
                               │
                    n_lyso par image → Results_with_true.csv
```

**Sorties** (dans `dossier/output/`) :

| Fichier | Description |
|---|---|
| `Results_with_true.csv` | Statistiques de fluorescence + `n_lyso` par image |
| `*_best.png` | Meilleur plan z par image |
| `*_mask_overlay.png` | Image + masque soma |
| `*_annotated.png` | Image + ROI lysosomes annotées |

**Paramètres clés :**

| Paramètre | Défaut | Description |
|---|---|---|
| `cnn_threshold` | 0.23 | Seuil CNN (calibré via courbe Precision-Recall) |
| `diameter` | 8 | Diamètre lysosomes Cellpose (px) |
| `min_area` / `max_area` | 5 / 500 | Filtre aire des lysosomes (px²) |
| `distance_transform` | 40 | Seuil distance pour isoler le cœur du soma |

---

## `Tracking.ipynb` — Suivi temporel

Suit les lysosomes frame par frame dans un stack TIFF multi-frames.

```
Stack TIFF (T frames) → Tri chronologique → Détection par frame
                                                    │
                              ┌─────────────────────┴──────────────────────┐
                         flag_CNN=True                               flag_CNN=False
                    CNN + Cellpose + TrackPy                       TrackPy seul
                    (précis, focalisé somas)                      (rapide, global)
                              └─────────────────────┬──────────────────────┘
                                                    │
                                           7 figures d'analyse
```

**Figures générées** (dans `output_dir/`) :

| Figure | Description |
|---|---|
| `histogramme_trajectoires[_CNN].png` | Distribution des longueurs de trajectoires |
| `msd_fit[_CNN].png` | MSD moyen + exposant α de diffusion |
| `nombre_lysosomes_par_frame[_CNN].png` | Évolution temporelle du nombre de lysosomes |
| `trajectoires_moyenne[_CNN].png` | Trajectoires superposées sur image moyenne |
| `densite_lysosomes[_CNN].png` | Carte de densité spatiale (KDE) |
| `distribution_vitesses[_CNN].png` | Distribution des vitesses instantanées |
| `trajectoires_vitesse[_CNN].png` | Trajectoires colorées par vitesse |

**Interprétation de l'exposant α (MSD) :**

| Valeur | Régime | Signification biologique |
|---|---|---|
| α ≈ 1 | Diffusion brownienne | Mouvement libre non dirigé |
| α < 1 | Sous-diffusion | Lysosome confiné (cage cytosquelettique) |
| α > 1 | Supra-diffusion | Transport actif sur microtubules |

---

## Structure des dossiers recommandée

```
projet/
├── Include.ipynb
├── soma_cnn_test.pth               ← généré par Pipeline_ML
├── patches_seuil_0.3_64x64_float32.npy ← à créer dans Pipeline_ML ou à télécharger  
│
├── Fluo_Analyse/
│   ├── Fluo_Analyse.ipynb
│   └── README_Fluo_Analyse
│
├── Pipeline_ML/
│   ├── Pipeline_ML.ipynb
│   └── README_Pipeline_ML
│
├── Tracking/
│   ├── Tracking.ipynb
│   └── README_Tracking
│
├── data/
│   ├── images_fixes/               ← TIFFs pour Fluo_Analyse
│   └── stacks_temporels/           ← TIFFs multi-frames pour Tracking
│
└── output/
    ├── analyse/                    ← sorties Fluo_Analyse
    └── tracking/                   ← sorties Tracking
```

---

## Référence rapide des paramètres à calibrer

Ces paramètres ont le plus d'impact sur la qualité des résultats et
doivent être ajustés selon les caractéristiques de l'acquisition :

| Paramètre | Notebook(s) | Impact |
|---|---|---|
| `cnn_threshold` | Fluo_Analyse, Tracking | Sensibilité détection somas |
| `diameter_cellpose` | Fluo_Analyse, Tracking | Taille lysosomes attendue |
| `cellprob_threshold` | Fluo_Analyse, Tracking | Sensibilité Cellpose |
| `search_range` | Tracking | Portée de liaison inter-frames |
| `threshold_filtered` | Tracking | Longueur minimale trajectoire |
| `n_clusters` | Generation | Granularité du clustering |
