# 🧠 Cortex AI

# Multimodal Brain Tumor Clinical Decision Support System

**Graduation Project | Artificial Intelligence | Computer Vision | NLP | Multimodal Learning | Explainable AI**

---

## 📖 Overview

Cortex AI is an AI-powered clinical decision support system designed to assist healthcare professionals in brain tumor diagnosis by combining MRI image analysis with radiology report understanding.

The system integrates **Computer Vision**, **Natural Language Processing (NLP)**, **Multimodal Deep Learning**, and **Explainable AI (XAI)** to generate accurate, interpretable, and clinically meaningful diagnostic insights.

---

# 📌 Problem Statement

Brain tumor diagnosis depends on multiple sources of information such as MRI scans and radiology reports.

Traditional AI systems usually analyze only one modality, limiting diagnostic performance.

Cortex AI solves this problem by combining medical images and clinical text into one multimodal AI system capable of producing more comprehensive and explainable predictions.

---

# 🎯 Project Objectives

- Detect brain tumors from MRI scans
- Analyze radiology reports using NLP
- Extract imaging and textual features
- Fuse both modalities into one prediction model
- Generate explainable AI outputs
- Support clinicians with intelligent decision assistance

---

# 🚀 Key Features

## 🖼️ Computer Vision

- MRI preprocessing
- Image normalization
- Brain tumor segmentation
- Feature extraction
- Deep learning-based analysis

---

## 📝 Natural Language Processing

- Clinical report preprocessing
- Medical text cleaning
- BioBERT embeddings
- ClinicalBERT embeddings
- Clinical feature extraction

---

## 🔗 Multimodal Fusion

- Image feature fusion
- Text feature fusion
- Joint multimodal representation
- Deep learning classification
- Brain tumor prediction

---

## 🔍 Explainable AI (XAI)

- Grad-CAM visualization
- SHAP explanations
- Feature importance analysis
- Transparent model interpretation

---

## 🌐 Interactive Dashboard

- Streamlit interface
- MRI upload
- Report upload
- Prediction visualization
- Explainability visualization

---

# 🏗️ System Architecture

The Cortex AI pipeline consists of four main modules.

## Computer Vision Module

Processes MRI scans and extracts imaging features.

↓

## NLP Module

Processes radiology reports and generates medical text embeddings.

↓

## Multimodal Fusion Module

Combines image and text representations for prediction.

↓

## Explainability & Interface Layer

Displays predictions with interpretable explanations.

---

# 🔄 High-Level Workflow

```text
MRI Images
      │
      ▼
Computer Vision Module
      │
      ▼
Image Features
                    \
                     \
                      ► Multimodal Fusion ► Prediction ► Explainability ► Dashboard
                     /
                    /
Text Reports
      │
      ▼
NLP Module
      │
      ▼
Text Features
```

---

# 📂 Repository Structure

```text
CortexAI/
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   ├── sample_data/
│   └── README.md
│
├── docs/
│
├── models/
│   ├── segmentation/
│   ├── nlp/
│   └── fusion/
│
├── notebooks/
│
├── reports/
│
├── src/
│   ├── cv_module/
│   ├── nlp_module/
│   ├── fusion_module/
│   ├── explainability/
│   ├── ui/
│   └── utils/
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# 🧰 Technology Stack

## Programming

- Python

---

## Computer Vision

- PyTorch
- MONAI
- OpenCV
- NumPy

---

## Natural Language Processing

- Hugging Face Transformers
- BioBERT
- ClinicalBERT

---

## Machine Learning

- Scikit-learn
- Pandas

---

## Explainable AI

- SHAP
- Grad-CAM

---

## User Interface

- Streamlit

---

# 📊 Datasets

Cortex AI uses two complementary datasets.

---

## 🖼️ Computer Vision Dataset

### BraTS2020 Training Dataset (Kaggle)

Used for:

- MRI preprocessing
- Brain tumor segmentation
- Feature extraction
- Deep learning model training

**Dataset Link**

https://www.kaggle.com/datasets/awsaf49/brats2020-training-data

Expected folder:

```text
datasets/raw/brats2020/
```

---

## 📝 NLP Dataset

### TextBraTS

Used for:

- Radiology report processing
- Medical text embeddings
- Clinical NLP
- Multimodal alignment

**Dataset Link**

https://github.com/Jupitern52/TextBraTS

Expected folder:

```text
datasets/raw/textbrats/
```

---

# ⚙️ Dataset Setup

After cloning the repository:

## Clone

```bash
git clone https://github.com/nour-hossam7/CortexAI.git
cd CortexAI
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

## Download Datasets

Download the datasets using the links above and place them in:

```text
datasets/raw/brats2020/
datasets/raw/textbrats/
```

---

## Verify Dataset Structure

Run:

```bash
python -m src.utils.setup_data
```

The script will automatically:

- verify dataset folders
- create missing directories
- validate processed folders
- ensure project readiness

---

# 🧪 Module Responsibilities

## Computer Vision

- MRI preprocessing
- Image normalization
- Tumor segmentation
- Feature extraction

---

## NLP

- Text preprocessing
- BioBERT embeddings
- ClinicalBERT embeddings
- Feature extraction

---

## Fusion

- Feature alignment
- Image-text fusion
- Multimodal learning
- Prediction generation

---

## Explainability

- Grad-CAM
- SHAP
- Interpretation

---

## UI

- Streamlit Dashboard
- Prediction interface
- Visualization
- Integration

---

# 👥 Team Members

| Name | Role |
|------|------|
| Nour Hossam | NLP Developer |
| Mariam Mohamed | Computer Vision Developer |
| Ammar Kamal | Fusion Module Developer |
| Ahmed Hossam | Explainable AI Developer |
| Ibrahim Mahmoud | UI & Integration Developer |

---

# 📅 Project Status

## Current Status

🚧 Under Development

### Completed

- Repository structure ✅
- Project architecture ✅
- Dataset organization ✅

### In Progress

- Computer Vision Module ⏳
- NLP Module ⏳
- Fusion Module ⏳
- Explainable AI ⏳
- Streamlit Dashboard ⏳

---

# 🗺️ Development Roadmap

### Phase 1

- Repository setup
- Environment setup
- Dataset organization

### Phase 2

- Computer Vision
- NLP

### Phase 3

- Multimodal Fusion

### Phase 4

- Explainable AI

### Phase 5

- Dashboard
- Evaluation
- Final Testing

---

# 🤝 Notes for Contributors

- Raw datasets are **NOT uploaded to GitHub** because they exceed GitHub's storage limits.
- Every team member must download the datasets locally.
- Keep the directory structure unchanged.
- Do not rename dataset folders.
- Store trained models inside the **models/** directory.
- Store evaluation outputs inside **reports/**.
- Refer to **datasets/README.md** for complete dataset setup instructions.

---

# 📜 License

This project is licensed under the MIT License.

---

# 📧 Contact

**Nour Hossam**

GitHub:

https://github.com/nour-hossam7
