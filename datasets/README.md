# Datasets Guide

This folder contains the dataset structure and dataset organization for **CortexAI**, an AI-powered multimodal system for brain tumor analysis using MRI images and radiology reports.

The project is built around two primary datasets:

* **BraTS2020 Training Dataset (Kaggle)** → Computer Vision (MRI image segmentation & feature extraction)
* **TextBraTS** → Natural Language Processing (radiology reports & clinical text analysis)

---

# Dataset Download Links

## Computer Vision Dataset

**BraTS2020 Training Dataset (Kaggle)**

https://www.kaggle.com/datasets/awsaf49/brats2020-training-data

---

## Natural Language Processing Dataset

**TextBraTS**

https://github.com/Jupitern52/TextBraTS

---

# Dataset Structure

```text
datasets/
│
├── raw/
│   ├── brats2020/
│   └── textbrats/
│
├── processed/
│   ├── cv/
│   ├── nlp/
│   └── fusion/
│
└── sample_data/
```

---

# Raw Datasets

## 1. BraTS2020 Training Dataset (Kaggle)

This folder stores the original MRI brain tumor dataset used by the **Computer Vision** module.

Expected location:

```text
datasets/raw/brats2020/
```

The original folder structure provided by Kaggle should remain unchanged.

This dataset is used for:

* MRI preprocessing
* Tumor segmentation
* Feature extraction
* Deep learning model training

---

## 2. TextBraTS

This folder stores the radiology reports and text-based clinical information used by the **NLP** module.

Expected location:

```text
datasets/raw/textbrats/
```

This dataset is used for:

* Text preprocessing
* Clinical information extraction
* Medical embeddings
* NLP model training

---

# Processed Data Folders

Preprocessing outputs should be saved inside the following folders.

### Computer Vision

```text
datasets/processed/cv/
```

Contains:

* Resized MRI images
* Normalized scans
* Segmentation masks
* Extracted image features

---

### Natural Language Processing

```text
datasets/processed/nlp/
```

Contains:

* Cleaned reports
* Tokenized text
* Medical embeddings
* NLP features

---

### Multimodal Fusion

```text
datasets/processed/fusion/
```

Contains:

* Combined CV + NLP features
* Final multimodal datasets
* Inputs used for fusion models

---

# Team Setup Instructions

After cloning the repository, every team member should follow these steps.

## Step 1

Install project dependencies.

```bash
pip install -r requirements.txt
```

---

## Step 2

Download both datasets using the links above.

Place them inside:

BraTS2020

```text
datasets/raw/brats2020/
```

TextBraTS

```text
datasets/raw/textbrats/
```

---

## Step 3

Verify the dataset structure.

From the project root directory run:

```bash
python -m src.utils.setup_data
```

The script will automatically:

* Create missing folders
* Verify BraTS2020 availability
* Verify TextBraTS availability
* Verify processed directories
* Confirm that the project structure is ready

---

# Current Dataset Mapping

## Computer Vision Module

Input dataset

```text
datasets/raw/brats2020/
```

Processed output

```text
datasets/processed/cv/
```

---

## NLP Module

Input dataset

```text
datasets/raw/textbrats/
```

Processed output

```text
datasets/processed/nlp/
```

---

## Fusion Module

Inputs

```text
datasets/processed/cv/
datasets/processed/nlp/
```

Output

```text
datasets/processed/fusion/
```

---

# Recommended Team Workflow

## Computer Vision Developer

* Load MRI scans from:

```text
datasets/raw/brats2020/
```

Tasks:

* Image preprocessing
* Tumor segmentation
* Feature extraction

Save outputs to:

```text
datasets/processed/cv/
```

---

## NLP Developer

Load reports from:

```text
datasets/raw/textbrats/
```

Tasks:

* Text cleaning
* Tokenization
* Medical embeddings
* Clinical feature extraction

Save outputs to:

```text
datasets/processed/nlp/
```

---

## Fusion Developer

Load processed data from:

```text
datasets/processed/cv/
datasets/processed/nlp/
```

Tasks:

* Feature alignment
* Multimodal fusion
* Dataset preparation
* Fusion model training

Save outputs to:

```text
datasets/processed/fusion/
```

---

# Notes

* Raw medical datasets are **NOT uploaded to GitHub** because they exceed GitHub's file size limits.
* Every team member must download the datasets locally.
* Keep the folder structure exactly the same for everyone.
* Do **NOT** rename folders.
* Do **NOT** modify dataset paths.
* Processed outputs should only be committed if they are required for experiments or demonstrations.
* Use `sample_data/` only for small demo files or testing.

---

# Standard Dataset Paths

All CortexAI modules expect the following directory structure.

```text
datasets/raw/brats2020/
datasets/raw/textbrats/

datasets/processed/cv/
datasets/processed/nlp/
datasets/processed/fusion/

datasets/sample_data/
```

---

# Important

Maintaining the same directory structure across all team members ensures that every module works consistently without requiring path modifications.

If you change folder names or locations, the project modules may fail to locate the required datasets.

## Important for Team Members

Do **NOT** upload the raw datasets to GitHub.

Only the folder structure is tracked in the repository using `.gitkeep` files.

Each team member must download the datasets locally using the links above and place them inside:

datasets/raw/brats2020/

datasets/raw/textbrats/

