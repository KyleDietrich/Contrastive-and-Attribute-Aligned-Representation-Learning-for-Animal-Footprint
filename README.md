# Footprint Representation Learning with Contrastive CNNs

Final project for **CSE 5526**  
Authors: Kyle Dietrich, Wonyoung Kim 

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
