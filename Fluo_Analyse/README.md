# Fluo_Analyse — Extractions informations & Comptage de lysosomes par image de microscopie

Ce notebook réalise l'**analyse en production** d'un lot d'images de microscopie
de fluorescence TIFF. Il s'appuie sur le CNN soma (`SomaCNN`) entraîné dans
`Pipeline_ML.ipynb` pour détecter les somas, puis utilise **Cellpose**
pour segmenter et compter les lysosomes à l'intérieur de chaque soma détecté.

---

## Prérequis

### Fichiers nécessaires
| Fichier | Rôle |
|---|---|
| `Include.ipynb` | Toutes les dépendances et fonctions utilitaires |
| `soma_cnn_test.pth` | Poids du CNN soma (généré par `Generation_ML_puncta.ipynb`) |

### Dépendances Python
```bash
pip install torch torchvision torchaudio
pip install scikit-image==0.25.2
pip install "numpy<2"
pip install cellpose==2.1.0
pip install tifffile tqdm matplotlib scipy pandas
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
        ├──► Cellpose_Analyse_Count                 → Comptage lysosomes / ROI
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

### CNN soma
| Paramètre | Défaut | Description |
|---|---|---|
| `cnn_threshold` | 0.23 | Seuil de probabilité CNN — calibré via la courbe Precision-Recall |
| `patch_size` | 64 | Doit correspondre à la valeur utilisée à l'entraînement |
| `overlap_thresh` | 0.9 | Fraction de chevauchement pour la fusion des ROI |

### Prétraitement image
| Paramètre | Défaut | Description |
|---|---|---|
| `pretraitement_sigma` | 1.5 | Sigma du flou gaussien |
| `clip_limit` | 0.02 | Limite CLAHE |
| `distance_transform` | 40 | Seuil distance pour isoler le cœur du soma |
| `margin` | 40 | Marge (px) autour du soma pour la suppression |

### Cellpose (lysosomes)
| Paramètre | Défaut | Description |
|---|---|---|
| `diameter` | 8 | Diamètre estimé des lysosomes (px) |
| `cellprob_threshold` | 0.1 | Seuil Cellpose (↓ = plus de détections) |
| `flow_threshold` | 0.5 | Cohérence des flux (↑ = plus strict) |
| `min_area` / `max_area` | 5 / 500 | Filtre aire des objets détectés (px²) |
| `min_circularity` | 0.5 | Filtre circularité (élimine artefacts allongés) |
| `max_axis_ratio` | 2 | Rapport axes max (élimine dendrites) |
| `top_hat_radius` | 10 | Rayon disque top-hat (> rayon lysosome) |

---

## Fichiers générés

Tous les fichiers de sortie sont placés dans `dossier/output/` :

| Fichier | Contenu |
|---|---|
| `<nom>_best.png` | Image du meilleur plan z |
| `<nom>_mask_overlay.png` | Image + masque soma superposé |
| `<nom>_annotated.png` | Image + ROI lysosomes colorées par probabilité CNN |
| `Results_with_true.csv` | Tableau complet : statistiques de fluorescence + `n_lyso` par image |

### Colonnes du CSV
| Colonne | Description |
|---|---|
| `file` | Nom du fichier source |
| `area` | Aire totale de l'image (px) |
| `pmean` / `pmin` / `pmax` | Statistiques d'intensité brutes |
| `area_true` | Aire hors-soma (px) |
| `pmean_true` / `pmin_true` / `pmax_true` | Statistiques d'intensité hors-soma |
| `n_lyso` | Nombre total de lysosomes détectés |
| `canal` | Canal de fluorescence (`w1TRITC`, `w2GFP`, ou `autre`) |
