# Footprint Representation Learning with Contrastive CNNs

Final project for **CSE 5526**  
Authors: **Kyle Dietrich**, **Wonyoung Kim**

---

## 1. Project Overview

This project explores **representation learning for animal footprints** using the **AnimalClue** footprint dataset.

Our main goals are:

- Learn **footprint image embeddings** with **contrastive learning** (SimCLR-style, NT-Xent loss).
- Evaluate those embeddings via a **linear probe** for species classification.
- Compare contrastive pretraining to a **fully supervised CNN** trained end-to-end.
- Explore similarity-based species identification using **k-NN retrieval** in the learned embedding space.

Rather than just training a standard classifier, we focus on learning a **general-purpose footprint representation** that can support multiple downstream tasks.

---

## 2. Data: AnimalClue Footprint Dataset

We use the **`footprint_yolo`** dataset from AnimalClue (Hugging Face gated dataset). You must request access from the dataset page and clone/download it manually.

Expected YOLO-format structure (after download):

```text
dataset/
  footprint_yolo/
    species/
      train/
        images/
        labels/
      valid/
        images/
        labels/
      test/
        images/
        labels/
```

- Images are in `images/*.jpg`
- YOLO label files are in `labels/*.txt` with contents:

  ```text
  class_id cx cy w h
  ```

  (normalized coordinates, possibly multiple lines per file).

### YOLO → Patch Dataset

We convert the detection dataset into a **patch-level classification dataset**:

For each labeled bounding box:

1. Convert YOLO box `(cx, cy, w, h)` to pixel coordinates.
2. Optionally expand the box by a small margin for context.
3. Crop the footprint patch from the image.
4. Save it into a species-specific subdirectory.

Resulting structure:

```text
dataset/
  footprint_patches/
    train/
      species_000/
        *.jpg
      species_001/
        *.jpg
      ...
    valid/
      species_000/
      ...
    test/
      species_000/
      ...
    class_names.txt
```

We infer **117 species classes** directly from the label files (there is no `data.yaml` with class names in the dataset). `class_names.txt` stores a reproducible mapping from numeric class IDs to generic labels like `species_000`, `species_001`, etc.

---

## 3. Repository Structure

Core project files:

- `prep_footprint_patches.ipynb`  
  Build the patch-level classification dataset from YOLO annotations.

- `contrastive_main.ipynb`  
  Train a **contrastive CNN encoder** (SimCLR-style) on footprint patches and save `footprint_encoder_contrastive.pth`.

- `linear_probe.ipynb`  
  - Load the frozen contrastive encoder.  
  - Precompute embeddings for train/valid splits.  
  - Train a **linear classifier** on top of the embeddings.  
  - Report classification performance.

- `supervised_baseline.ipynb` (in progress)  
  Train the same CNN architecture **end-to-end** in a fully supervised way and compare its accuracy to the contrastive + linear probe setup.

- `README.md` (this file)  
  Project description and run instructions.

---

## 4. Environment & Dependencies

The project was developed on **Windows** with **Python 3.14** and **CPU-only PyTorch** (no GPU). A more typical setup (Python 3.10–3.12 + CUDA) should also work.

### Required packages

Install at least:

```bash
pip install torch torchvision pillow pyyaml tqdm matplotlib
```

---

## 5. How to Run

### 5.1 Step 1 – Preprocess YOLO → Patches

Notebook: **`prep_footprint_patches.ipynb`**

1. Open the notebook.
2. At the top, set the paths to match your local setup, for example:

   ```python
   from pathlib import Path

   YOLO_ROOT = Path(
       r"C:\path\to\dataset\footprint_yolo\species"
   )
   PATCH_ROOT = Path(
       r"C:\path\to\dataset\footprint_patches"
   )
   ```

3. Run the notebook top to bottom. It will:
   - Scan label files to infer the set of species class IDs.
   - Create a `class_names.txt` mapping (e.g., `species_000` → class 0).
   - Crop footprint patches and save them in:

     ```text
     footprint_patches/{train,valid,test}/species_XXX/*.jpg
     ```

After this step, you should have a **patch-level classification dataset** ready for both contrastive and supervised experiments.

---

### 5.2 Step 2 – Contrastive Pretraining (SimCLR-style)

Notebook: **`contrastive_main.ipynb`**

1. Set `PATCH_ROOT` at the top:

   ```python
   from pathlib import Path

   PATCH_ROOT = Path(
       r"C:\path\to\dataset\footprint_patches"
   )
   image_size = 128
   batch_size = 64
   ```

2. Make sure `FootprintPatchDataset` uses the `train` split under `PATCH_ROOT` and that `num_workers=0` in your `DataLoader`.

3. Run cells that:
   - Define `FootprintEncoder` (CNN encoder → 256-dim embedding).
   - Define the projection head (256 → 128).
   - Set up strong augmentations and `ContrastiveTransform`.
   - Implement the NT-Xent loss (`nt_xent_loss`).
   - Implement `train_contrastive`.

4. Call `main()` (or the training function) to run contrastive training.  
   - Training runs on CPU by default.  
   - After training, the encoder weights are saved as:

     ```text
     footprint_encoder_contrastive.pth
     ```

---

### 5.3 Step 3 – Linear Probe

Notebook: **`linear_probe.ipynb`**

This notebook evaluates how good the learned embeddings are for species classification.

1. Set `PATCH_ROOT` and `ENCODER_CKPT` near the top:

   ```python
   from pathlib import Path

   PATCH_ROOT = Path(
       r"C:\path\to\dataset\footprint_patches"
   )
   ENCODER_CKPT = "footprint_encoder_contrastive.pth"
   image_size = 128
   batch_size = 64
   ```

2. Load the patch datasets (train/valid) with simple transforms (e.g., `Resize` + `ToTensor`).

3. Load the frozen encoder:

4. **Precompute embeddings**:

   ```python
   train_feats, train_labels = compute_features(encoder, train_loader, device)
   val_feats, val_labels = compute_features(encoder, val_loader, device)
   ```

   This runs the encoder once over all patches and stores 256-dim features in memory.

5. Create feature datasets and loaders, then train the linear classifier:

   ```python
   model = LinearClassifier(feature_dim=256, num_classes=num_classes).to(device)
   train_linear_probe(...)
   ```

6. The notebook will print training and validation accuracy per epoch.  

---

### 5.4 Step 4 – Supervised Baseline (in progress)

Notebook: **`supervised_baseline.ipynb`**

Planned workflow:

1. Use the same `FootprintPatchDataset` as above (`PATCH_ROOT/{train,valid,test}`).
2. Define `SupervisedCNN`:

   ```python
   class SupervisedCNN(nn.Module):
       def __init__(self, feature_dim=256, num_classes=117):
           super().__init__()
           self.encoder = FootprintEncoder(feature_dim=feature_dim)
           self.fc = nn.Linear(feature_dim, num_classes)

       def forward(self, x):
           feats = self.encoder(x)
           logits = self.fc(feats)
           return logits
   ```

3. Train end-to-end with cross-entropy on the train split, validate on the valid split, and finally evaluate on the test split.
4. Compare **supervised accuracy** with the **contrastive + linear probe** accuracy reported from `linear_probe.ipynb`.

This will answer the main question:  
> Does contrastive pretraining help learn better footprint representations than training a supervised classifier directly?

---

## 6. Notes & Limitations

- Training is currently configured for **CPU-only** due to environment constraints; GPU training is recommended if available.
- Class names are generic (`species_000`, `species_001`, …) because the dataset does not expose human-readable species labels.
- Some notebook code includes minor compatibility workarounds for **Python 3.14 + CPU PyTorch**; in a more standard Python/PyTorch setup, these may not be necessary.

---
