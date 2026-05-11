# Tracking Lysosomes — Suivi temporel par microscopie de fluorescence

Notebook de **suivi temporel de lysosomes** dans un stack TIFF multi-frames.
Combine un CNN de détection de somas, Cellpose pour la segmentation des lysosomes,
et TrackPy pour le linking des trajectoires frame par frame.

---

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

### Fichiers nécessaires
| Fichier | Rôle |
|---|---|
| `Include.ipynb` | Toutes les dépendances et fonctions utilitaires |
| `soma_cnn_test.pth` | Poids du CNN soma (généré par `Pipeline_ML.ipynb`) |


## Pipeline

```
Fichiers TIFF (_t001, _t002, …)
        │
        ▼
Tri numérique + chargement → stack (T, H, W)
        │
        ├── flag_CNN=False ──────────────────────────────────────────────────┐
        │                                                                    │
        │   TrackPy seul                                                     │
        │   (détection sur image entière, rapide)                           │
        │                                                                    │
        └── flag_CNN=True ───────────────────────────────────────────────────┤
                                                                             │
            Frame 0 uniquement :                                             │
            CNN_Patches_Construction → Clusterization → Finale_Fusion       │
            (détection des ROI somas, fixes pour tout le stack)             │
                                                                             │
            Pour chaque frame :                                              │
            Cellpose_Analyse_Track_GPU → DataFrame lysosomes                │
                                                                             │
        ◄────────────────────────────────────────────────────────────────────┘
        │
        ▼
TrackPy link_df → trajectoires
        │
        ▼
filter_stubs (longueur min : threshold_filtered frames)
        │
        ▼
7 figures sauvegardées dans output_dir
```

---

## Deux modes de détection

### `flag_CNN=True` — Pipeline complet (recommandé)
- Détection des ROI somas sur la **première frame uniquement** (les somas ne bougent pas)
- Segmentation Cellpose des lysosomes **dans chaque ROI** pour toutes les frames
- Plus précis, focalisé sur les lysosomes intra-somas
- `search_range` recommandé : **2 px**

### `flag_CNN=False` — TrackPy seul (rapide)
- Détection TrackPy sur l'image entière à chaque frame
- Plus rapide, adapté aux stacks très longs ou aux tests de paramètres
- `search_range` recommandé : **4 px**

---

## Paramètres clés

### CNN soma
| Paramètre | Défaut | Description |
|---|---|---|
| `cnn_threshold` | 0.23 | Seuil probabilité CNN |
| `patch_size` | 64 | Taille patch (doit correspondre à l'entraînement) |
| `cnn_batch_size` | 512 | Patches par passe CNN (↓ si erreur OOM GPU) |
| `overlap_thresh` | 0.1 | Seuil fusion ROI |

### Cellpose
| Paramètre | Défaut | Description |
|---|---|---|
| `diameter_cellpose` | 11 | Diamètre lysosomes (px) |
| `cellprob_threshold` | 0.1 | Seuil Cellpose |
| `flow_threshold` | 0.5 | Cohérence des flux |
| `min_area` / `max_area` | 5 / 500 | Filtre aire (px²) |
| `min_circularity` | 0.5 | Filtre circularité |
| `max_axis_ratio` | 2 | Filtre dendrites |
| `top_hat_radius` | 10 | Rayon top-hat |

### TrackPy
| Paramètre | Défaut | Description |
|---|---|---|
| `diameter_trackpy` | 11 | Diamètre particules (impair) |
| `search_range` | 4 | Portée recherche entre frames (px) — mettre 2 avec CNN |
| `memory` | 4 | Frames de mémorisation avant abandon |
| `threshold_filtered` | 20 | Longueur minimale trajectoire conservée (frames) |

---

## Figures générées

Toutes sauvegardées dans `output_dir`, avec suffixe `_CNN` si `flag_CNN=True` :

| Fichier | Description |
|---|---|
| `histogramme_trajectoires.png` | Distribution des longueurs de trajectoires |
| `msd_fit.png` | MSD moyen + fit loi de puissance (exposant α) |
| `nombre_lysosomes_par_frame.png` | Évolution temporelle du nombre de lysosomes |
| `trajectoires_moyenne.png` | Trajectoires sur image moyenne du stack |
| `densite_lysosomes.png` | Carte de densité spatiale KDE |
| `distribution_vitesses.png` | Distribution des vitesses instantanées |
| `trajectoires_vitesse.png` | Trajectoires colorées par vitesse |

### Interprétation de l'exposant α (MSD)

```
α ≈ 1   → diffusion brownienne normale
α < 1   → diffusion sous-diffusive (lysosome confiné, ex. cage cytosquelettique)
α > 1   → diffusion supra-diffusive (transport actif sur microtubules)
```

---

## Convention de nommage des fichiers TIFF

Le notebook attend des fichiers nommés avec un suffixe `_t<N>` :

```
CTRL 1 _ mitotracker_lysotracker_001_t001.tif
CTRL 1 _ mitotracker_lysotracker_001_t002.tif
...
```

Le tri est **numérique** (extrait l'entier N) pour éviter l'ordre
alphabétique incorrect (`_t1, _t10, _t100, _t2…`).
Adapter le motif `glob` si la convention de nommage est différente.
