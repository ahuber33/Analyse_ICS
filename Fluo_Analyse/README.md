# Fluo_Analyse — Extractions informations & Comptage de punctas par image de microscopie

Ce notebook réalise l'**analyse en production** d'un lot d'images de microscopie
de fluorescence TIFF. Il s'appuie sur le CNN soma (`SomaCNN`) entraîné dans
`Pipeline_ML.ipynb` pour détecter les somas, puis utilise **Cellpose**
pour segmenter et compter les punctas à l'intérieur de chaque soma détecté.

---

## Prérequis

### Fichiers nécessaires
| Fichier | Rôle |
|---|---|
| `Include.ipynb` | Toutes les dépendances et fonctions utilitaires |
| `soma_cnn_test.pth` | Poids du CNN soma (généré par `Pipeline_ML.ipynb`) |

## Prérequis

```bash
pip install torch torchvision torchaudio
pip install scikit-image==0.25.2
pip install tifffile==2023.4.12
pip install scipy==1.17.1
pip install numpy==1.26.4
pip install pandas==2.2.2
pip install umap-learn
pip install trackpy
pip install cellpose==2.1.0
pip install tqdm matplotlib
```

> Un GPU CUDA est recommandé (×5 à ×20 plus rapide qu'en CPU).

---

## Pipeline par image

```
Fichier TIFF (z-stack)
        │
        ▼
Best_plan_determination        → Plan le plus net (variance max)
        │
        ├──► Extract_fluo_informations              → Stats brutes (aire, moy, min, max)
        │
        ├──► Extract_fluo_informations_without_nucleus → Stats hors-soma + masque
        │
        ├──► CNN_Patches_Construction               → Détection somas (LoG + CNN)
        │         │
        │         ▼
        │    Clusterization_detected_patches        → Fusion patches chevauchants
        │         │
        │         ▼
        │    Finale_Fusion_patches                  → ROI finales
        │         │
        │         ▼
        ├──► Cellpose_Analyse_Count                 → Comptage punctas / ROI
        │
        ├──► Save_images_and_overlay                → Images annotées sur disque
        ├──► Save_final_patches                     → ROI avec probabilités CNN
        └──► Creation_CSV                           → Export résultats
```

---

## Modes d'utilisation

### Mode batch (cellule 6)
Traite automatiquement tous les fichiers `.tif` d'un dossier et exporte un CSV.

```python
fichiers = list(dossier.glob("*.tif"))[:10]  # 🔧 retirer [:10] pour tout traiter
```

### Mode fichier individuel (cellule 7)
Analyse un seul fichier avec visualisation activée, utile pour calibrer les paramètres.

```python
fichier = "Ctrl1_Tubulin+pTau_011_w1TRITC.tif"  # 🔧 à modifier
img_best = Best_plan_determination(chemin, output_flag=True)
```

---

## Paramètres clés à calibrer

### Pre-processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pretraitement_sigma` | `1.5` | Gaussian blur sigma before blob detection |
| `clip_limit` | `0.02` | CLAHE contrast enhancement clip limit |

### Soma detection (CNN)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cnn_threshold` | `0.23` | Minimum CNN probability to classify a patch as soma |
| `patch_size` | `64` | Size (px) of the square patch extracted around each blob |
| `cnn_batch_size` | `256` | Patches per CNN forward pass (increase for faster processing) |

### Soma exclusion (fluorescence stats)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `distance_transform` | `40` | Distance (px) to erode soma mask before exclusion |
| `margin` | `40` | Extra margin (px) added around each soma bounding box |
| `default_threshold` | `400` | Threshold applied by default to extract fluo informations |

### ROI fusion

| Parameter | Default | Description |
|-----------|---------|-------------|
| `overlap_thresh` | `0.9` | Fraction of overlap above which a smaller ROI is discarded |

### Cellpose segmentation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `diameter` | `8` | Expected puncta diameter in pixels |
| `cellprob_threshold` | `0.1` | Cellpose cell probability threshold |
| `flow_threshold` | `0.5` | Cellpose flow error threshold |
| `min_area` | `5` | Minimum area (px²) to keep a detected object |
| `max_area` | `500` | Maximum area (px²) to keep a detected object |
| `min_circularity` | `0.5` | Minimum circularity (0–1) — filters elongated dendrite fragments |
| `max_axis_ratio` | `2` | Maximum ratio of major/minor axis — filters dendrites |
| `top_hat_radius` | `10` | Disk radius for white top-hat pre-filter |
| `cellpose_batch_size` | `8` | Patches per Cellpose forward pass |

---

## Fichiers générés

Tous les fichiers de sortie sont placés dans `dossier/output/` :

| Fichier | Contenu |
|---|---|
| `<nom>_best.png` | Image du meilleur plan z |
| `<nom>_mask_overlay.png` | Image + masque soma superposé |
| `<nom>_annotated.png` | Image + ROI punctas colorées par probabilité CNN |
| `Results_with_true.csv` | Tableau complet : statistiques de fluorescence + `n_lyso` par image |

### Colonnes du CSV
| Colonne | Description |
|---|---|
| `file` | Nom du fichier source |
| `area` | Aire totale de l'image (px) |
| `pmean` / `pmin` / `pmax` | Statistiques d'intensité brutes |
| `area_true` | Aire hors-soma (px) |
| `pmean_true` / `pmin_true` / `pmax_true` | Statistiques d'intensité hors-soma |
| `n_lyso` | Nombre total de punctas détectés |
| `canal` | Canal de fluorescence (`w1TRITC`, `w2GFP`, ou `autre`) |