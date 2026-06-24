# 🧠 Cortex AI

### Multimodal Brain Tumor Clinical Decision Support System

**Graduation Project | Artificial Intelligence | Medical Imaging | NLP | Explainable AI**

Cortex AI is an AI-powered **clinical decision support system** designed to assist healthcare professionals in **brain tumor diagnosis** by combining **MRI image analysis** with **radiology report understanding**.

The system integrates **Computer Vision**, **Natural Language Processing (NLP)**, **Multimodal Learning**, and **Explainable AI (XAI)** to provide **accurate, interpretable, and clinically meaningful diagnostic insights**.

---

# 📌 Problem Statement

Brain tumor diagnosis often relies on multiple sources of information, including **MRI scans** and **radiology reports**. Analyzing these sources separately can lead to fragmented understanding, delayed decision-making, and reduced diagnostic confidence.

**Cortex AI** addresses this challenge by integrating **medical imaging data** and **textual clinical information** into a unified intelligent system capable of supporting diagnostic decisions in a more comprehensive and explainable manner.

---

# 🎯 Project Objectives

* Detect and analyze brain tumors from **MRI scans**
* Extract clinically relevant information from **radiology reports**
* Combine image and text features using **multimodal fusion**
* Provide **transparent and explainable** predictions
* Support clinicians with **AI-assisted diagnostic decision-making**

---

# 🚀 Key Features

## 🖼️ MRI Analysis

* MRI preprocessing and normalization
* Brain tumor segmentation
* Imaging feature extraction
* Tumor region visualization

## 📝 Clinical Report Analysis

* Medical text preprocessing
* Clinical report understanding
* **BioBERT** embeddings
* **ClinicalBERT** embeddings
* Extraction of clinically meaningful textual features

## 🔗 Multimodal Fusion

* Fusion of imaging and textual features
* Deep learning–based classification
* Unified multimodal prediction pipeline
* Clinical decision support output

## 🔍 Explainable AI (XAI)

* **Grad-CAM** visual explanations for MRI-based predictions
* **SHAP** analysis for feature importance
* Transparent interpretation of model behavior
* Increased trust and interpretability in clinical outputs

## 🌐 Interactive User Interface

* Streamlit-based dashboard
* MRI upload and analysis
* Report upload and processing
* Prediction and explanation visualization

---

# 🏗️ System Architecture

The Cortex AI pipeline consists of four major layers:

1. **Computer Vision Module**
   Processes MRI scans, performs tumor-related analysis, and extracts imaging features.

2. **NLP Module**
   Processes radiology reports and converts them into meaningful text embeddings using domain-specific medical language models.

3. **Fusion Module**
   Combines image-based and text-based representations into a unified multimodal feature space for final prediction.

4. **Explainability & Interface Layer**
   Generates explanations for predictions and exposes the system through an interactive clinical dashboard.

### High-Level Flow

```text
MRI Images
   ↓
Computer Vision Module
   ↓
Image Features
                     \
                      \
                       → Multimodal Fusion Module → Prediction → Explainability Layer → User Interface
                      /
                     /
Radiology Reports
   ↓
NLP Module
   ↓
Text Embeddings
```

---

# 📂 Repository Structure

```text
CortexAI/
│
├── datasets/
│   ├── raw/                  # Raw downloaded datasets
│   ├── processed/            # Cleaned / transformed datasets ready for training
│   └── sample_data/          # Small sample files for testing/demo
│
├── docs/
│   ├── architecture/         # Architecture diagrams and technical docs
│   ├── presentation/         # Slides / presentations
│   └── proposal/             # Proposal and project documentation
│
├── models/
│   ├── fusion/               # Saved multimodal/fusion models
│   ├── nlp/                  # Saved NLP models / checkpoints
│   └── segmentation/         # Saved CV / segmentation models
│
├── notebooks/                # Experiments and exploratory notebooks
│
├── reports/
│   ├── evaluation/           # Evaluation reports / metrics
│   ├── figures/              # Visual outputs, plots, diagrams
│   └── results/              # Final results and generated outputs
│
├── src/
│   ├── cv_module/            # Computer Vision pipeline
│   ├── explainability/       # Grad-CAM / SHAP / interpretation logic
│   ├── fusion_module/        # Multimodal fusion models and inference
│   ├── nlp_module/           # NLP preprocessing and embeddings
│   ├── ui/                   # Streamlit application
│   └── utils/                # Helper utilities, config, setup scripts
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

# 🧰 Technology Stack

## Programming Language

* Python

## Computer Vision & Medical Imaging

* PyTorch
* MONAI
* OpenCV
* NumPy

## Natural Language Processing

* Hugging Face Transformers
* BioBERT
* ClinicalBERT

## Machine Learning / Data Processing

* Scikit-learn
* Pandas

## Explainable AI

* SHAP
* Grad-CAM

## Deployment / Interface

* Streamlit
* Flask *(optional if used for backend APIs / integration)*

---

# 📊 Datasets

Cortex AI uses a **multimodal dataset setup** that combines **brain MRI data** with **clinical text data**.

## 1) MRI Dataset — BraTS2020

Used for the **Computer Vision module** to support:

* brain tumor analysis
* segmentation tasks
* image-based feature extraction

**Expected local folder:**

```bash
datasets/raw/brats2020/
```

## 2) Clinical Text Dataset — TextBraTS

Used for the **NLP module** and **Fusion module** to support:

* radiology report understanding
* text embedding generation
* image-text multimodal alignment

**Expected local folder:**

```bash
datasets/raw/textbrats/
```

---

# 🗂️ Dataset Directory Structure

Each team member should keep the same local dataset structure:

```text
datasets/
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

# ⚙️ Dataset Setup Guide

To make the repository ready for all team members without confusion, follow the steps below **after cloning the repository**.

## Step 1 — Clone the repository

```bash
git clone https://github.com/nour-hossam7/CortexAI.git
cd CortexAI
```

## Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

## Step 3 — Download the datasets manually

### BraTS2020

Download the MRI dataset and place it inside:

```bash
datasets/raw/brats2020/
```

### TextBraTS

Download the text dataset and place it inside:

```bash
datasets/raw/textbrats/
```

## Step 4 — Keep processed folders ready

The processed outputs of each module should be stored in:

```bash
datasets/processed/cv/
datasets/processed/nlp/
datasets/processed/fusion/
```

## Step 5 — Run dataset setup / verification script

After the raw datasets are placed correctly, run:

```bash
python src/utils/setup_data.py
```

This script is expected to:

* verify that required dataset folders exist
* check whether BraTS2020 data is available
* check whether TextBraTS data is available
* ensure processed folders are ready for use

---

# 🧪 Module Responsibilities

## Computer Vision Module

Responsible for:

* loading MRI scans
* preprocessing and normalization
* tumor-related segmentation / imaging analysis
* feature extraction from MRI volumes

## NLP Module

Responsible for:

* preprocessing radiology reports
* extracting embeddings using BioBERT / ClinicalBERT
* preparing text features for multimodal fusion

## Fusion Module

Responsible for:

* combining image and text representations
* training the multimodal classifier
* producing the final prediction output

## Explainability Module

Responsible for:

* generating Grad-CAM visualizations
* performing SHAP-based interpretation
* supporting transparent model analysis

## UI / Integration Module

Responsible for:

* building the Streamlit dashboard
* connecting CV, NLP, Fusion, and XAI outputs
* displaying predictions and explanations interactively

---

# 👥 Team Members

| Name                | Role                       |
| ------------------- | -------------------------- |
| **Nour Hossam**     | NLP Developer              |
| **Mariam Mohamed**  | Computer Vision Developer  |
| **Ammar Kamal**     | Fusion Module Developer    |
| **Ahmed Hossam**    | Explainable AI Developer   |
| **Ibrahim Mahmoud** | UI & Integration Developer |

---

# 📅 Project Status

## 🚧 Current Status

**Under Development**

## Current Progress

* Repository setup ✅
* Project architecture design ✅
* Dataset preparation in progress ⏳
* Module implementation in progress ⏳

## Planned Development Phases

### Phase 1 — Project Foundation

* Repository setup
* Folder structure organization
* Dataset preparation
* Environment setup

### Phase 2 — Core Modeling Modules

* Computer Vision module
* NLP module

### Phase 3 — Multimodal Fusion

* Image-text feature fusion
* Joint multimodal training

### Phase 4 — Explainable AI

* Grad-CAM integration
* SHAP analysis
* model interpretation layer

### Phase 5 — Deployment & Evaluation

* Streamlit dashboard integration
* system testing
* evaluation and reporting

---

# 📝 Notes for Contributors / Team Members

* **Raw medical datasets are not uploaded to GitHub** because of storage size and dataset distribution constraints.
* Every team member must keep the **same local dataset structure** to avoid broken paths.
* Large trained models and heavy outputs should be saved in the appropriate folders under `models/` and `reports/`.
* If a script depends on local dataset paths, it should always follow the standard project structure defined in this README.

---

# 📜 License

This project is licensed under the **MIT License**.
See the `LICENSE` file for more details.

---

# 📧 Contact

For questions, collaborations, or contributions:

**Nour Hossam**
GitHub: https://github.com/nour-hossam7
