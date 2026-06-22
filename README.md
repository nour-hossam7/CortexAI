# 🧠 Cortex AI
### Multimodal Brain Tumor Clinical Decision Support System

> Graduation Project | Artificial Intelligence | Medical Imaging | NLP | Explainable AI

Cortex AI is an AI-powered clinical decision support system designed to assist healthcare professionals in brain tumor diagnosis by combining MRI image analysis and radiology report understanding.

The system leverages Computer Vision, Natural Language Processing (NLP), Multimodal Learning, and Explainable AI to provide accurate, interpretable, and reliable diagnostic insights.

---

# 📌 Problem Statement

Brain tumor diagnosis often relies on multiple sources of information, including MRI scans and radiology reports. Analyzing these sources separately may lead to incomplete understanding and delayed decision-making.

Cortex AI addresses this challenge by integrating medical imaging data and textual clinical information into a unified intelligent system capable of supporting diagnostic decisions.

---

# 🎯 Project Objectives

- Detect and analyze brain tumors from MRI scans.
- Extract meaningful information from radiology reports.
- Combine image and text features using Multimodal Fusion.
- Provide transparent and explainable predictions.
- Support clinicians with AI-assisted decision-making.

---

# 🚀 Key Features

### 🖼️ MRI Analysis
- MRI preprocessing
- Tumor segmentation
- Feature extraction
- Visualization

### 📝 Clinical Report Analysis
- Medical text preprocessing
- BioBERT embeddings
- ClinicalBERT embeddings
- Clinical information extraction

### 🔗 Multimodal Fusion
- Fusion of imaging and textual features
- Deep learning classification model
- Clinical decision support

### 🔍 Explainable AI (XAI)
- Grad-CAM visual explanations
- SHAP feature importance analysis
- Transparent model interpretation

### 🌐 Interactive User Interface
- Streamlit-based dashboard
- MRI upload and analysis
- Report upload and processing
- Prediction visualization

---

# 🏗️ System Architecture

MRI Images
↓
Computer Vision Module
↓
Image Features

Radiology Reports
↓
NLP Module
↓
Text Embeddings

Image Features + Text Embeddings
↓
Multimodal Fusion Module
↓
Prediction
↓
Explainability Layer
↓
User Interface

---

# 📂 Repository Structure

```text
CortexAI
│
├── datasets
│   ├── raw
│   ├── processed
│   └── sample_data
│
├── docs
│   ├── proposal
│   ├── architecture
│   └── presentation
│
├── models
│   ├── segmentation
│   ├── nlp
│   └── fusion
│
├── reports
│   ├── evaluation
│   ├── figures
│   └── results
│
├── src
│   ├── cv_module
│   ├── nlp_module
│   ├── fusion_module
│   ├── explainability
│   ├── ui
│   └── utils
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

# 🧰 Technologies Used

### Programming Language
- Python

### Computer Vision
- PyTorch
- MONAI
- OpenCV

### Natural Language Processing
- Transformers
- BioBERT
- ClinicalBERT

### Machine Learning
- Scikit-Learn
- NumPy
- Pandas

### Explainable AI
- SHAP
- Grad-CAM

### Deployment
- Streamlit
- Flask

---

# 📊 Datasets

### MRI Dataset
- BraTS Dataset

### Clinical Reports
- MIMIC-CXR Reports

### Additional Medical Resources
- PubMed

---

# 👥 Team Members

| Name | Role |
|--------|--------|
| Nour Hossam | NLP Developer |
| Mariam Mohamed | Computer Vision Developer |
| Ammar Kamal | Fusion Module Developer |
| Ahmed Hossam | Explainable AI Developer |
| Ibrahim Mahmoud | UI & Integration Developer |

---

# 📅 Project Status

🚧 Currently Under Development

Phase 1:
- Repository Setup ✅
- Project Architecture ✅
- Dataset Preparation ⏳

Phase 2:
- Computer Vision Module
- NLP Module

Phase 3:
- Multimodal Fusion

Phase 4:
- Explainable AI

Phase 5:
- Deployment & Evaluation

---

# 📜 License

This project is licensed under the MIT License.

---

# 📧 Contact

For questions, collaborations, or contributions:

**Nour Hossam**

GitHub: https://github.com/nour-hossam7
