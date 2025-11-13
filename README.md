# Footprint Representation Learning with Contrastive CNNs

Final project for **CSE 5526**  
Authors: Kyle Dietrich
         Wonyoung Kim 

---

## 1. Project Overview

This project explores **representation learning for animal footprints** using the **AnimalClue** dataset.  
The main goals are:

- Learn **footprint image embeddings** with **contrastive learning** (SimCLR-style, NT-Xent loss).
- Evaluate those embeddings via a **linear probe** for species classification.
- Compare contrastive pretraining to a **fully supervised CNN baseline** (planned).

Rather than training a standard softmax classifier end-to-end, the project focuses on learning a **general-purpose footprint representation** that can be reused for downstream tasks.

---

## 2. Data: AnimalClue Footprint Subset

The project uses the **footprint subset** of the AnimalClue dataset in **YOLO format** (`footprint_yolo` on Hugging Face).

Original format:

- Images and YOLO labels:  
  - `images/{train,val,test}/*.jpg`  
  - `labels/{train,val,test}/*.txt`
- Each label file contains one or more bounding boxes:  
  `class_id cx cy w h` (normalized coordinates).

### YOLO → Patch Dataset

Because the original dataset is built for **detection**, the first step is to convert it into a **patch-level classification dataset**:

- For each footprint bounding box:
  - Convert YOLO box `(cx, cy, w, h)` to pixel coordinates.
  - Optionally expand the box by a small margin (context).
  - Crop the patch and save it under a class-specific folder.

Final structure:

```text
footprint_patches/
  train/
    species_A/
      *.jpg
    species_B/
      *.jpg
    ...
  val/
    ...
  test/
    ...
```

This patch dataset is then used both for **contrastive pretraining** and **classification**.

---

## 3. Environment & Dependencies

Recommended:

- **Python** ≥ 3.9  
- **PyTorch** ≥ 2.0  
- **CUDA** GPU (optional but strongly recommended)

Key packages:

- `torch`, `torchvision`
- `Pillow`
- `pyyaml`
- `tqdm`
- `matplotlib` (for plotting in `linear_probe.ipynb`)
---

## 4. Workflow

### Step 1 – Preprocess YOLO → Patches

Notebook: **`prep_footprint_patches.ipynb`**

1. Set these paths near the top of the notebook:
   - `YOLO_ROOT = Path("path/to/footprint_yolo")`
   - `PATCH_ROOT = Path("path/to/footprint_patches")`
2. Run all cells to:
   - Load class names from `data.yaml`.
   - Convert YOLO bounding boxes into cropped patches.
   - Save patches in `PATCH_ROOT/{train,val,test}/<class_name>/*.jpg`.

At the end, `footprint_patches/` should contain image patches grouped by species.

---

### Step 2 – Contrastive Pretraining (SimCLR-style)

Notebook: **`contrastive_main.ipynb`**

CNN architecture (high level):

- 4 convolutional blocks with BatchNorm, ReLU, and MaxPool / AdaptiveAvgPool.
- Final fully connected layer → 256-dim embedding.

Contrastive loss:

- SimCLR-style NT-Xent with temperature (e.g., τ = 0.5).
- For a batch of N images, each with 2 views → 2N embeddings.
- Each view’s positive is its paired view; all others are negatives.

---

### Step 3 – Linear Probe

Notebook: **`linear_probe.ipynb`**

Goal: **evaluate the quality of the learned embeddings** for species classification.

This gives a **linear probe performance number** to compare against a fully supervised CNN.

---

### Step 4 – Supervised Baseline (Planned)

Notebook: **`supervised_baseline.ipynb`** (to be implemented)

Planned steps:

1. Use the same `FootprintEncoder` architecture but **train end-to-end** on the patch dataset with cross-entropy.
2. Evaluate on the validation set.
3. Compare:

- **Supervised-from-scratch accuracy** vs  
- **Linear probe accuracy on contrastive encoder**

This comparison will answer:

> Does contrastive pretraining help learn better footprint representations than training a supervised classifier directly?

---

## 5. Limitations 

---
