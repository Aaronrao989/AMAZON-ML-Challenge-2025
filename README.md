# 🧠 Smart Product Pricing — Multimodal ML Project

<div align="center">

**An AI-powered pricing model that predicts product prices by analyzing both textual and visual information**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B.svg)](https://streamlit.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-ML-yellow.svg)](https://catboost.ai)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📘 Overview

This project builds an **AI-powered pricing model** that predicts product prices by analyzing both **textual** and **visual** information. It integrates product descriptions, catalog text, and image embeddings to create a unified multimodal feature space.

### 🎯 Key Components

The solution combines:

- 📝 **Text-based TF-IDF features** — Captures semantic meaning from product descriptions
- 📊 **Statistical text features** — Word/character-level metrics for linguistic analysis
- 🖼️ **Precomputed image embeddings** — Visual features extracted via deep learning
- 🤖 **CatBoost regression model** — Robust gradient boosting for price prediction
- 📈 **Streamlit dashboard** — Interactive visualization and real-time inference

---

## 🧩 Folder Structure

```
student_resource/
├── 📱 app.py                          # Streamlit Dashboard
├── 📂 src/
│   ├── data_preprocessing.py          # Cleans and merges raw dataset
│   └── text_features.py               # Builds TF-IDF and text statistics
├── 📂 embeddings/
│   ├── image_features.py              # Extracts image embeddings (EfficientNet)
│   ├── model_training.py              # Trains CatBoost model
│   ├── model_inference.py             # Generates predictions on CSV
│   └── generate_submission.py         # Prepares submission.csv for test set
├── 📂 artifacts/
│   ├── cleaned_data.csv               # Processed training data
│   ├── tfidf_vectorizer.pkl           # Saved TF-IDF vectorizer
│   ├── catboost_model.pkl             # Trained model (CatBoostRegressor)
│   ├── image_embeddings.npy           # Image embedding array
│   ├── image_embeddings.csv           # Sample ID alignment for embeddings
│   ├── predictions.csv                # Predictions on training/validation
│   └── submission.csv                 # Final submission output
├── 📂 dataset/
│   ├── train.csv
│   ├── sample_test_out.csv
│   ├── sample_test.csv
│   └── test.csv
├── 📂 images/                         # Raw image files (img_12345.jpg etc.)
└── 📄 README.md                       # Documentation
```

---

## 🚀 Features

### 📝 Text Processing
- **TF-IDF vectorization** — `ngram_range=(1,2)`, max_features=4000–10000
- **Statistical text features** — word count, char count, average word length, unique words

### 🖼️ Image Feature Extraction
- **EfficientNetB0** pretrained on ImageNet
- **Embedding dimension** = 1280

### 🔗 Multimodal Fusion
- Concatenates `[TF-IDF | text stats | image embeddings]`

### 🤖 Model
- `CatBoostRegressor(iterations=800, depth=8, learning_rate=0.05, loss_function="MAE")`

### 📊 Evaluation Metrics
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **SMAPE** (Symmetric Mean Absolute Percentage Error)

### 🎨 Interactive Dashboard
- Data exploration and feature visualization
- Real-time predictions and submission generation

---

## ⚙️ Environment Setup

### 1️⃣ Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

**Or manually:**

```bash
pip install streamlit pandas numpy joblib pillow matplotlib altair scipy catboost
```

> **💡 Apple Silicon Users (M1/M2/M3):**  
> If facing issues with CatBoost:
> ```bash
> conda install -c conda-forge catboost
> ```

---

## 🧪 Training Workflow

### Step 1️⃣ Data Cleaning

```bash
python src/data_preprocessing.py
```

**Outputs:**
- `artifacts/cleaned_data.csv`

---

### Step 2️⃣ Text Feature Extraction

```bash
python src/text_features.py
```

**Outputs:**
- `artifacts/tfidf_vectorizer.pkl`
- `artifacts/text_features.csv`

---

### Step 3️⃣ Image Embedding Extraction

```bash
python src/image_features.py
```

**Outputs:**
- `artifacts/image_embeddings.npy`
- `artifacts/image_embeddings.csv`

---

### Step 4️⃣ Model Training

```bash
python src/model_training.py
```

**Outputs:**
- `artifacts/catboost_model.pkl`

#### 📊 Performance Metrics

| Metric | Cross-Validation (avg) | Holdout Set |
|--------|------------------------|-------------|
| **MAE**    | ~14.93 ± 1.00      | 5.58        |
| **R²**     | 0.12 ± 0.03        | 0.56        |

---

## 🧮 Inference and Submission

### Predict on cleaned training data

```bash
python src/model_inference.py
```

**→ Saves:** `artifacts/predictions.csv`

---

### Generate final submission file

```bash
python src/generate_submission.py
```

**→ Saves:** `artifacts/submission.csv`

---

## 🖥️ Streamlit App

### Run the dashboard

```bash
python -m streamlit run app.py
```

### ✨ Features

✅ Load cleaned dataset and inspect columns  
✅ Compute text statistics  
✅ Visualize TF-IDF vocabulary and IDF values  
✅ Preview image embeddings and local images  
✅ Load saved CatBoost model (`.pkl` or `.cbm`)  
✅ Run inference and save predictions  
✅ Generate submission files directly  
✅ Download artifacts (model, vectorizer, submission)

---

## 🧠 ML Approach Summary

### 🔹 Data Modality Fusion

The model combines three feature sources:

1. **Textual information** → TF-IDF + statistical text features
2. **Image embeddings** → 1280-dim feature vector from EfficientNet
3. **Numeric metadata** (e.g., IPQ)

**Final feature vector per sample:**

```
[ TF-IDF (4000–10000 dims) + text stats (5) + image embeddings (1280) ]
```

---

### 🔹 Model

1. **CatBoost Regressor** chosen for its superior handling of heterogeneous and sparse data
2. Trained using **5-fold cross-validation**
3. **Early stopping** used to avoid overfitting

---

### 🔹 Experiments & Observations

1. Text-only models achieved **R² ≈ 0.12**
2. Adding image embeddings increased stability and improved **holdout R² ≈ 0.55**
3. Feature scaling was not necessary due to CatBoost's internal normalization

---

### 🔹 Conclusion

> **Combining multimodal features significantly improves generalization.**  
> TF-IDF remains a strong baseline for product descriptions, and image embeddings enhance contextual understanding.

---

## 🧰 Artifacts to Share

| Artifact               | Description                               |
|------------------------|-------------------------------------------|
| `cleaned_data.csv`     | Cleaned and merged dataset                |
| `tfidf_vectorizer.pkl` | Trained TF-IDF model                      |
| `catboost_model.pkl`   | Final regression model                    |
| `image_embeddings.npy` | Extracted visual feature matrix           |
| `submission.csv`       | Final predictions for test data           |
| `app.py`               | Streamlit dashboard for demo and analysis |

---

## 👨‍💻 Contributors

**Team:** Student Resource — Smart Pricing  
**Lead Developer:** Aaron Rao  
**Technologies:** Python, Pandas, CatBoost, TensorFlow, Streamlit, NumPy, SciPy, Matplotlib

---

## 📄 License

This project is open-source under the **MIT License**.

---

## 🏁 Quick Start Summary

```bash
# Activate environment
source .venv/bin/activate

# Train Model
python src/model_training.py

# Launch Streamlit App
python -m streamlit run app.py

# Generate Predictions
python src/generate_submission.py
```

---

<div align="center">

**Made with ❤️ by the Smart Product Pricing Team**

</div>