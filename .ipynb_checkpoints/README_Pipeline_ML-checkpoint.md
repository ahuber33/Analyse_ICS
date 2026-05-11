# Pipeline ML — Détection et classification de puncta (somas)

Pipeline d'apprentissage automatique non supervisé puis supervisé pour la détection de structures ponctuelles (somas / puncta) dans des images de microscopie de fluorescence TIFF.

---

## Vue d'ensemble

```
Images TIFF  →  Extraction patches  →  Auto-encodeur  →  UMAP + KMeans
                                                               ↓
                                              Inspection visuelle des clusters
                                                               ↓
                                           Sélection manuelle (PatchSelector)
                                                               ↓
                                              Entraînement CNN (SomaCNN)
                                                               ↓
                                           Évaluation + Sauvegarde du modèle
```

---

## Fichiers

| Fichier | Rôle |
|---|---|
| `Include.ipynb` | Dépendances, classes et fonctions utilitaires (imports, `ConvAutoEncoder`, `SomaCNN`, `SomaBinaryDataset`, fonctions de détection, de clustering, de tracking…) |
| `Generation_ML_puncta.ipynb` | Notebook principal — construction du dataset, entraînement, évaluation |

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

> Un GPU CUDA est recommandé pour l'inférence CNN et Cellpose, mais le pipeline fonctionne sur CPU.

---

## Étapes du pipeline

### 1. Construction du dataset (`optionnel`)
Parcourt récursivement un dossier de fichiers TIFF, sélectionne le plan de meilleure netteté, détecte les blobs (LoG) et extrait des patches 64×64 px.

```python
patches, metadata = build_dataset_specific_from_folder(root_folder, patch_size=64)
np.save("patches_64x64_float32.npy", patches)
```

### 2. Chargement et sous-échantillonnage
```python
patches = np.load("patches_seuil_0.3_64x64_float32.npy")
N = 30000  # ajustable
```

### 3. Entraînement de l'auto-encodeur
`ConvAutoEncoder` apprend une représentation latente compacte (encodeur 3 couches Conv, décodeur 3 couches ConvTranspose, critère MSE).

### 4. Clustering non supervisé
- Extraction des embeddings (stratégie **GAP** ou **Flatten**)
- Réduction dimensionnelle **UMAP** → 2D
- Clustering **KMeans** (`n_clusters` ajustable, défaut 20)

### 5. Sélection manuelle (`PatchSelector`)
Interface graphique interactive (backend `matplotlib tk`) pour valider visuellement les patches positifs (somas) :

| Touche | Action |
|---|---|
| Clic gauche | Sélectionner / désélectionner un patch |
| `→` / `←` | Naviguer entre les blocs |
| `S` | Sauvegarder la sélection |
| `Q` | Quitter |

### 6. Entraînement du classificateur (`SomaCNN`)
CNN binaire léger (3 blocs Conv + AdaptiveAvgPool + Linear + Sigmoid).  
Entraîné avec `neg_ratio=10` pour compenser le déséquilibre des classes.

### 7. Évaluation
- Courbe **Precision-Recall**
- Seuil optimal **F1**
- Seuil à **précision contrainte** (≥ 90 %)

### 8. Sauvegarde
```python
torch.save(model.state_dict(), "soma_cnn_test.pth")
```

Rechargement :
```python
model = SomaCNN()
model.load_state_dict(torch.load("soma_cnn_test.pth"))
model.eval()
```

---

## Paramètres clés

| Paramètre | Défaut | Description |
|---|---|---|
| `patch_size` | 64 | Taille des patches extraits (px) |
| `N` | 30 000 | Taille du sous-échantillon pour l'AE |
| `n_clusters` | 20 | Nombre de clusters KMeans |
| `epochs` | 10 | Epochs d'entraînement du CNN |
| `neg_ratio` | 10 | Ratio négatifs / positifs |
| `target_precision` | 0.9 | Contrainte de précision pour le seuil |

---

## Fichiers générés

| Fichier | Contenu |
|---|---|
| `patches_64x64_float32.npy` | Dataset de patches bruts |
| `manual_soma_selection.npy` | Indices des patches validés manuellement |
| `manual_soma_patches.npy` | Patches positifs validés |
| `soma_cnn_test.pth` | Poids du CNN entraîné |
