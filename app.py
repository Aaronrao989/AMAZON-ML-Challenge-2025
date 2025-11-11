"""
Streamlit dashboard for the Smart Product Pricing project (robust for Streamlit Cloud).
Includes a safe "Generate model" handler with full traceback display and fallback training.

Usage:
- Place artifacts in artifacts/ or set ARTIFACT_URLS in Streamlit Secrets
- Deploy on Streamlit Cloud or run locally:
    streamlit run app.py
"""

import os
import io
import traceback
from pathlib import Path
import time

import numpy as np
import pandas as pd
import joblib
from PIL import Image
import streamlit as st
import requests

# ML imports (sklearn fallback)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix, hstack as sparse_hstack

# Page config
st.set_page_config(layout="wide", page_title="Smart Pricing — Project Explorer")

# ---------------------------
# Helper: download missing artifacts (Streamlit-secrets driven)
# ---------------------------
def download_if_missing(url, dest_path, chunk_size=1 << 20, timeout=60):
    dest = Path(dest_path)
    if dest.exists():
        return True
    if not url:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with st.spinner(f"Downloading {dest.name} ..."):
            resp = requests.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0) or 0)
            downloaded = 0
            tmp_path = str(dest_path) + ".part"
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            st.progress(min(1.0, downloaded / total))
            os.replace(tmp_path, dest_path)
        return dest.exists()
    except Exception as e:
        st.error(f"Failed to download {dest.name}: {e}")
        return False

# ---------------------------
# Sidebar: defaults + uploaders
# ---------------------------
st.sidebar.title("Config / Artifacts")

DEFAULTS = {
    "DATA_PATH": "artifacts/cleaned_data.csv",
    "TEST_CSV": "dataset/test.csv",
    "TFIDF_PATH": "artifacts/tfidf_vectorizer.pkl",
    "MODEL_PATH": "artifacts/catboost_model.pkl",
    "IMG_NPY": "artifacts/image_embeddings.npy",
    "IMG_CSV": "artifacts/image_embeddings.csv",
    "IMAGES_DIR": "images",
    "SUBMISSION_OUT": "artifacts/submission.csv"
}

DATA_PATH = st.sidebar.text_input("Cleaned data CSV", value=DEFAULTS["DATA_PATH"])
TEST_CSV = st.sidebar.text_input("Test CSV (raw)", value=DEFAULTS["TEST_CSV"])
TFIDF_PATH = st.sidebar.text_input("TF-IDF vectorizer", value=DEFAULTS["TFIDF_PATH"])
MODEL_PATH = st.sidebar.text_input("CatBoost model", value=DEFAULTS["MODEL_PATH"])
IMG_NPY = st.sidebar.text_input("Image embeddings (.npy)", value=DEFAULTS["IMG_NPY"])
IMG_CSV = st.sidebar.text_input("Image embeddings (.csv)", value=DEFAULTS["IMG_CSV"])
IMAGES_DIR = st.sidebar.text_input("Local images folder", value=DEFAULTS["IMAGES_DIR"])
SUBMISSION_OUT = st.sidebar.text_input("Submission output", value=DEFAULTS["SUBMISSION_OUT"])
MAX_IMAGE_PREVIEW = st.sidebar.slider("Max images to preview", 1, 16, 6)

st.sidebar.markdown("---")
st.sidebar.markdown("Built for: Smart Product Pricing (TF-IDF + Image embeddings + CatBoost)")

uploaded_model = st.sidebar.file_uploader("Upload model (.pkl / .cbm / .bin / .model)", type=["pkl", "cbm", "bin", "model"])
uploaded_tfidf = st.sidebar.file_uploader("Upload TF-IDF vectorizer (.pkl)", type=["pkl"])
uploaded_npy = st.sidebar.file_uploader("Upload image embeddings (.npy)", type=["npy"])
uploaded_img_csv = st.sidebar.file_uploader("Upload image embeddings CSV (optional)", type=["csv"])
uploaded_test_csv = st.sidebar.file_uploader("Upload test.csv (optional)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("Model generation options")
train_on_subset = st.sidebar.checkbox("Train on subset (safe for Cloud)", value=True)
subset_size = st.sidebar.number_input("Subset size (rows)", min_value=100, max_value=100000, value=2000, step=100)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)
catboost_params = st.sidebar.text_area("CatBoost params (JSON)", value='{"iterations":100, "verbose":0}', height=80)

st.sidebar.markdown("---")
st.sidebar.markdown("If artifacts are missing, set direct URLs in Streamlit Secrets as `ARTIFACT_URLS` (see README).")

# ---------------------------
# Auto-download artifacts via secrets
# ---------------------------
urls = {}
try:
    urls = st.secrets.get("ARTIFACT_URLS", {}) if hasattr(st, "secrets") else {}
except Exception:
    urls = {}

TFIDF_LOCAL = Path(TFIDF_PATH)
MODEL_LOCAL = Path(MODEL_PATH)
IMG_NPY_LOCAL = Path(IMG_NPY)
IMG_CSV_LOCAL = Path(IMG_CSV)
TEST_CSV_LOCAL = Path(TEST_CSV)

if not TFIDF_LOCAL.exists():
    download_if_missing(urls.get("tfidf"), TFIDF_LOCAL)
if not MODEL_LOCAL.exists():
    download_if_missing(urls.get("model"), MODEL_LOCAL)
if not IMG_NPY_LOCAL.exists():
    download_if_missing(urls.get("img_npy"), IMG_NPY_LOCAL)
if not IMG_CSV_LOCAL.exists():
    download_if_missing(urls.get("img_csv"), IMG_CSV_LOCAL)
if not TEST_CSV_LOCAL.exists():
    download_if_missing(urls.get("test_csv"), TEST_CSV_LOCAL)

# If user uploaded artifacts during session, persist them to tmp and override paths
if uploaded_model is not None:
    tmp_model_path = Path("tmp_uploaded_model" + Path(uploaded_model.name).suffix)
    with open(tmp_model_path, "wb") as f:
        f.write(uploaded_model.read())
    MODEL_PATH = str(tmp_model_path)
    MODEL_LOCAL = tmp_model_path

if uploaded_tfidf is not None:
    tmp_tfidf = Path("tmp_uploaded_tfidf.pkl")
    with open(tmp_tfidf, "wb") as f:
        f.write(uploaded_tfidf.read())
    TFIDF_PATH = str(tmp_tfidf)
    TFIDF_LOCAL = tmp_tfidf

if uploaded_npy is not None:
    tmp_npy = Path("tmp_uploaded_image_embeddings.npy")
    with open(tmp_npy, "wb") as f:
        f.write(uploaded_npy.read())
    IMG_NPY = str(tmp_npy)
    IMG_NPY_LOCAL = tmp_npy

if uploaded_img_csv is not None:
    tmp_img_csv = Path("tmp_uploaded_image_embeddings.csv")
    with open(tmp_img_csv, "wb") as f:
        f.write(uploaded_img_csv.read())
    IMG_CSV = str(tmp_img_csv)
    IMG_CSV_LOCAL = tmp_img_csv

if uploaded_test_csv is not None:
    tmp_test_csv = Path("tmp_uploaded_test.csv")
    with open(tmp_test_csv, "wb") as f:
        f.write(uploaded_test_csv.read())
    TEST_CSV = str(tmp_test_csv)
    TEST_CSV_LOCAL = tmp_test_csv

# ---------------------------
# Caching helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_vectorizer(path):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_npy(path):
    return np.load(path)

@st.cache_resource(show_spinner=False)
def load_model(path):
    last_err = None
    try:
        return joblib.load(path)
    except Exception as e_joblib:
        last_err = e_joblib
    try:
        from catboost import CatBoostRegressor, CatBoostClassifier
        try:
            m = CatBoostRegressor()
            m.load_model(path)
            return m
        except Exception:
            try:
                m2 = CatBoostClassifier()
                m2.load_model(path)
                return m2
            except Exception as e_cb2:
                last_err = e_cb2
    except Exception as e_imp:
        last_err = e_imp
    raise RuntimeError(f"Failed to load model with joblib or CatBoost. Last error:\n{last_err}")

# ---------------------------
# Small utilities
# ---------------------------
def basic_text_stats(s: pd.Series):
    s = s.fillna("").astype(str)
    n_words = s.str.split().apply(lambda ws: len(ws) if isinstance(ws, list) else 0)
    n_unique = s.str.split().apply(lambda ws: len(set(ws)) if isinstance(ws, list) else 0)
    return pd.DataFrame({
        "n_chars": s.str.len(),
        "n_words": n_words,
        "n_unique_words": n_unique
    })

def safe_text_series(df, col_name):
    if col_name in df.columns:
        return df[col_name].fillna("").astype(str)
    else:
        return pd.Series([""] * len(df), index=df.index)

def show_image_grid(image_paths, cols=3, size=(180, 180)):
    cols_w = st.columns(cols)
    for i, path in enumerate(image_paths):
        col = cols_w[i % cols]
        try:
            img = Image.open(path)
            col.image(img, caption=Path(path).name, use_column_width=True)
        except Exception as e:
            col.write(f"Error loading image: {e}")

def safe_tfidf_info(tf):
    info = {}
    try:
        info["vocab_size"] = len(tf.get_feature_names_out())
    except Exception:
        try:
            info["vocab_size"] = len(tf.get_feature_names())
        except Exception:
            info["vocab_size"] = None
    try:
        info["idf"] = pd.Series(tf.idf_, index=tf.get_feature_names_out()).sort_values()
    except Exception:
        info["idf"] = None
    return info

# ---------------------------
# Main UI
# ---------------------------
st.title("Smart Product Pricing — Project Explorer")
st.markdown("Visualize dataset, features, images and run inference using your saved model.")

# 1) Data load & preview
st.header("1) Data")
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Load cleaned dataset")
    uploaded = st.file_uploader("Upload cleaned_data.csv (optional)", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success("Loaded uploaded CSV")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
            df = None
    else:
        if os.path.exists(DATA_PATH):
            try:
                df = load_csv(DATA_PATH)
                st.success(f"Loaded {DATA_PATH}")
            except Exception as e:
                st.error(f"Failed to load {DATA_PATH}: {e}")
                df = None
        else:
            st.warning(f"{DATA_PATH} not found. Upload file to proceed.")
            df = None

    if df is not None:
        st.write(f"Rows: **{len(df)}**, Columns: **{len(df.columns)}**")
        st.dataframe(df.head(200))

with col2:
    st.subheader("Quick text stats")
    if df is not None:
        text_col = "combined_text" if "combined_text" in df.columns else ("catalog_content" if "catalog_content" in df.columns else (df.columns[0] if len(df.columns) > 0 else None))
        st.write("Using text column:", text_col)
        if text_col is not None:
            stats = basic_text_stats(df[text_col])
            st.write(stats.describe().T)
            st.bar_chart(stats["n_words"].clip(0, 100).value_counts().sort_index().head(50))
        else:
            st.write("No columns available to compute text stats.")

# 2) TF-IDF
st.header("2) Text features (TF-IDF)")
tf_loaded = False
tf = None
if os.path.exists(TFIDF_PATH):
    try:
        tf = load_vectorizer(TFIDF_PATH)
        tf_loaded = True
    except Exception as e:
        st.error(f"Failed to load TF-IDF vectorizer: {e}")
else:
    st.info("TF-IDF vectorizer not found at configured path.")

if tf_loaded:
    st.subheader("TF-IDF summary")
    info = safe_tfidf_info(tf)
    n_feats = info.get("vocab_size")
    if n_feats is not None:
        st.write(f"Vocabulary size: **{n_feats}**")
    else:
        st.write("Vocabulary size: unknown (sklearn version mismatch?)")
    top_k = st.slider("Top K TF-IDF tokens to show", 5, 200, 25)
    if info.get("idf") is not None:
        st.write("Top tokens (lowest idf → very common):")
        st.write(info["idf"].head(top_k))
    else:
        st.write("Could not extract idf (older sklearn or custom vectorizer).")

# 3) Image embeddings
st.header("3) Image features")
img_emb_loaded = False
img_emb = None
img_df = None
if os.path.exists(IMG_NPY):
    try:
        img_emb = load_npy(IMG_NPY)
        img_emb_loaded = True
        st.write("Loaded image embeddings:", IMG_NPY, "shape:", getattr(img_emb, "shape", None))
    except Exception as e:
        st.error(f"Failed to load .npy embeddings: {e}")
else:
    st.info("No image .npy found at configured path.")

if os.path.exists(IMG_CSV):
    try:
        img_df = pd.read_csv(IMG_CSV)
        st.write("Loaded image embeddings CSV (for sample-id alignment):", IMG_CSV)
    except Exception:
        img_df = None
else:
    img_df = None

if img_emb_loaded:
    st.subheader("Embedding details")
    try:
        st.write("Embedding dim:", img_emb.shape[1])
        st.write("Preview first rows (numeric):")
        st.dataframe(pd.DataFrame(img_emb[:10, :10], columns=[f"dim_{i}" for i in range(min(10, img_emb.shape[1]))]))
    except Exception:
        st.write("Could not preview embeddings (unexpected shape).")

    st.subheader("Preview images (from local folder)")
    if os.path.exists(IMAGES_DIR):
        if img_df is not None and "sample_id" in img_df.columns:
            preview_ids = img_df["sample_id"].head(MAX_IMAGE_PREVIEW).astype(str).tolist()
            picked = []
            for sid in preview_ids:
                fname = os.path.join(IMAGES_DIR, f"img_{sid}.jpg")
                if os.path.exists(fname):
                    picked.append(fname)
            if not picked:
                picked = [str(x) for x in list(Path(IMAGES_DIR).glob("*"))[:MAX_IMAGE_PREVIEW]]
            if picked:
                show_image_grid(picked, cols=3)
            else:
                st.write("No matching images found in images dir.")
        else:
            files = sorted(Path(IMAGES_DIR).glob("*"))[:MAX_IMAGE_PREVIEW]
            if files:
                show_image_grid([str(x) for x in files], cols=3)
            else:
                st.write("No images present in folder.")
    else:
        st.write("Images dir not found:", IMAGES_DIR)

# ---------------------------
# 4) Generate Model (robust)
# ---------------------------
st.header("4) Generate model (train & save)")
st.markdown("Train a new model from the cleaned dataset. Use `Train on subset` for quick tests on Streamlit Cloud.")

generate_clicked = st.button("Generate model")
if generate_clicked:
    try:
        # Basic checks
        if df is None:
            st.error("No cleaned dataset loaded. Upload or set DATA_PATH to proceed.")
        else:
            # Build dataset (combined_text)
            df_train = df.copy().fillna("")
            if "combined_text" not in df_train.columns:
                cat_series = safe_text_series(df_train, "catalog_content")
                text_series = safe_text_series(df_train, "text")
                df_train["combined_text"] = (cat_series + " " + text_series).str.lower()

            # Optionally sample subset for cloud safety
            if train_on_subset:
                n = min(int(subset_size), len(df_train))
                st.info(f"Training on subset: {n} rows (random_state={random_state})")
                df_train = df_train.sample(n=n, random_state=int(random_state)).reset_index(drop=True)
            else:
                st.info(f"Training on full dataset: {len(df_train)} rows")

            # Require TF-IDF loaded
            if not tf_loaded:
                st.error("TF-IDF vectorizer not loaded. Upload or configure TFIDF_PATH.")
                raise RuntimeError("TF-IDF missing")

            # Transform text features (sparse)
            X_text_sp = tf.transform(df_train["combined_text"].astype(str))

            # Numeric features: compute basic ones if missing
            num_cols = ["ipq", "word_count", "char_count", "avg_word_len", "num_unique_words"]
            if "word_count" not in df_train.columns:
                st.info("Computing text stats for numeric columns...")
                stats = basic_text_stats(df_train["combined_text"])
                for c in stats.columns:
                    df_train[c] = stats[c]

            for c in num_cols:
                if c not in df_train.columns:
                    df_train[c] = 0

            X_num = csr_matrix(df_train[num_cols].astype(float).values)

            # Image embeddings (optional)
            use_image = False
            if img_emb_loaded and img_df is not None and "sample_id" in img_df.columns:
                try:
                    emb_dim = img_emb.shape[1] if img_emb.ndim == 2 else None
                except Exception:
                    emb_dim = None
                if emb_dim:
                    emb_map = dict(zip(img_df["sample_id"].astype(str), range(len(img_df))))
                    emb_rows = []
                    for sid in df_train["sample_id"].astype(str):
                        idx = emb_map.get(sid, None)
                        if idx is None:
                            emb_rows.append(np.zeros(emb_dim, dtype=float))
                        else:
                            if idx < len(img_emb):
                                emb_rows.append(img_emb[idx])
                            else:
                                emb_rows.append(np.zeros(emb_dim, dtype=float))
                    X_img = csr_matrix(np.vstack(emb_rows))
                    use_image = True

            X_full = sparse_hstack([X_text_sp, X_num, X_img], format="csr") if use_image else sparse_hstack([X_text_sp, X_num], format="csr")

            st.write("Feature matrix shape:", X_full.shape)

            # Target column: try 'price' or 'target' or prompt user
            target_col = "price" if "price" in df_train.columns else ("target" if "target" in df_train.columns else None)
            if target_col is None:
                st.error("Target column not found (expected 'price' or 'target'). Add it to your cleaned dataset.")
                raise RuntimeError("Target missing")
            y = df_train[target_col].astype(float).values

            # Train/test split for quick eval
            X_tr, X_val, y_tr, y_val = train_test_split(X_full, y, test_size=0.1, random_state=int(random_state))

            # Try to train CatBoost if available; otherwise fallback to RandomForest
            model_obj = None
            used_catboost = False
            try:
                from catboost import CatBoostRegressor
                import json
                params = {}
                try:
                    params = json.loads(catboost_params)
                except Exception:
                    st.warning("CatBoost params JSON invalid, using defaults")
                    params = {"iterations": 100, "verbose": 0}
                st.info("Training CatBoostRegressor (native). This may take time...")
                # CatBoost can accept numpy arrays; convert if sparse
                X_tr_dense = X_tr.toarray() if hasattr(X_tr, "toarray") else X_tr
                X_val_dense = X_val.toarray() if hasattr(X_val, "toarray") else X_val
                m = CatBoostRegressor(**params)
                m.fit(X_tr_dense, y_tr, eval_set=(X_val_dense, y_val), verbose=params.get("verbose", 0))
                model_obj = m
                used_catboost = True
            except Exception as cb_exc:
                st.warning(f"CatBoost unavailable or failed: {cb_exc}. Falling back to RandomForestRegressor.")
                st.info("Training RandomForestRegressor (sklearn) — faster but less accurate.")
                rf = RandomForestRegressor(n_estimators=100, random_state=int(random_state), n_jobs=2)
                rf.fit(X_tr, y_tr)
                model_obj = rf

            # Evaluate quickly
            try:
                preds_val = model_obj.predict(X_val.toarray() if used_catboost and hasattr(X_val, "toarray") else X_val)
                rmse = mean_squared_error(y_val, preds_val, squared=False)
                st.success(f"Validation RMSE: {rmse:.4f}")
            except Exception:
                st.warning("Could not compute validation metric (shape mismatch?)")

            # Save model to artifacts/
            os.makedirs("artifacts", exist_ok=True)
            timestamp = int(time.time())
            # Save joblib pickle
            pkl_path = f"artifacts/model_{timestamp}.pkl"
            joblib.dump(model_obj, pkl_path, compress=3)
            st.success(f"Saved model pickle → {pkl_path}")
            # If CatBoost used, also save native cbm
            if used_catboost:
                try:
                    cbm_path = f"artifacts/catboost_model_{timestamp}.cbm"
                    model_obj.save_model(cbm_path)
                    st.success(f"Saved CatBoost native model → {cbm_path}")
                except Exception as e_savecb:
                    st.warning(f"Failed to save CatBoost native model: {e_savecb}")

            # Update sidebar MODEL_PATH to last saved pickle so inference uses it
            st.info("Updating MODEL_PATH to use the freshly saved model for inference.")
            st.sidebar.text_input("CatBoost model", value=pkl_path, key="MODEL_PATH_AFTER_SAVE")
            st.success("Model generation finished successfully.")

    except Exception as e:
        st.error(f"Model generation failed: {e}")
        with st.expander("Show full traceback"):
            st.text(traceback.format_exc())
        print("=== Model generation error ===")
        traceback.print_exc()

# ---------------------------
# 5) Model & Inference (load and predict)
# ---------------------------
st.header("5) Load model & run inference")
model_loaded = False
model = None
model_source_path = MODEL_PATH
# If user uploaded model, it's already overridden earlier
if os.path.exists(model_source_path):
    try:
        model = load_model(model_source_path)
        model_loaded = True
        st.success(f"Loaded model from: {model_source_path}")
    except Exception as e:
        st.error("Could not load model. See details below.")
        st.text(str(e))
        st.text(traceback.format_exc())
        st.info("If this is a CatBoost model, ensure 'catboost' is in your requirements or upload a joblib-pickled model.")
else:
    st.info(f"Model not found at configured path: {model_source_path}")
    st.info("You can update MODEL_PATH in the sidebar or upload a model file (pkl/cbm) using the sidebar uploader.")

run_infer = st.button("Run inference on cleaned_data (predict & save)")
if run_infer:
    if not model_loaded:
        st.error("Model not loaded. Check MODEL_PATH or upload a model in the sidebar.")
    elif df is None:
        st.error("No cleaned data loaded.")
    else:
        with st.spinner("Preprocessing and predicting..."):
            try:
                df_local = df.copy().fillna("")
                if "combined_text" not in df_local.columns:
                    cat_series = safe_text_series(df_local, "catalog_content")
                    text_series = safe_text_series(df_local, "text")
                    df_local["combined_text"] = (cat_series + " " + text_series).str.lower()

                if not tf_loaded:
                    st.error("TF-IDF not loaded; cannot transform text.")
                else:
                    X_text_sp = tf.transform(df_local["combined_text"].astype(str))
                    num_cols = ["ipq", "word_count", "char_count", "avg_word_len", "num_unique_words"]
                    if "word_count" not in df_local.columns:
                        stats = basic_text_stats(df_local["combined_text"])
                        for c in stats.columns:
                            df_local[c] = stats[c]
                    for c in num_cols:
                        if c not in df_local.columns:
                            df_local[c] = 0
                    X_num = csr_matrix(df_local[num_cols].astype(float).values)

                    use_image = False
                    if img_emb_loaded and img_df is not None and "sample_id" in img_df.columns:
                        try:
                            emb_dim = img_emb.shape[1] if img_emb.ndim == 2 else None
                        except Exception:
                            emb_dim = None
                        if emb_dim:
                            emb_map = dict(zip(img_df["sample_id"].astype(str), range(len(img_df))))
                            emb_rows = []
                            for sid in df_local["sample_id"].astype(str):
                                idx = emb_map.get(sid, None)
                                if idx is None:
                                    emb_rows.append(np.zeros(emb_dim, dtype=float))
                                else:
                                    if idx < len(img_emb):
                                        emb_rows.append(img_emb[idx])
                                    else:
                                        emb_rows.append(np.zeros(emb_dim, dtype=float))
                            X_img = csr_matrix(np.vstack(emb_rows))
                            use_image = True

                    X_full = sparse_hstack([X_text_sp, X_num, X_img], format="csr") if use_image else sparse_hstack([X_text_sp, X_num], format="csr")

                    st.write("Feature matrix shape (sparse):", X_full.shape)
                    preds = None
                    try:
                        preds = model.predict(X_full.toarray() if hasattr(X_full, "toarray") and hasattr(model, "predict") and getattr(model, "__class__", None).__name__.startswith("CatBoost") else X_full)
                    except Exception as e:
                        st.error("Model prediction failed (shape/feature-order mismatch).")
                        st.error(str(e))
                        st.text(traceback.format_exc())

                    if preds is not None:
                        df_out = df_local.copy()
                        df_out["predicted_price"] = preds
                        out_path = SUBMISSION_OUT if SUBMISSION_OUT else "artifacts/predictions.csv"
                        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                        if "sample_id" in df_out.columns:
                            df_out[["sample_id", "predicted_price"]].to_csv(out_path, index=False)
                        else:
                            df_out[["predicted_price"]].to_csv(out_path, index=False)
                        st.success(f"Predictions saved → {out_path}")
                        st.dataframe(df_out[["sample_id", "predicted_price"]].head(20) if "sample_id" in df_out.columns else df_out[["predicted_price"]].head(20))

            except Exception as e:
                st.error(f"Unexpected error during inference: {e}")
                st.text(traceback.format_exc())

# ---------------------------
# 6) Create submission from test.csv
# ---------------------------
st.header("6) Generate submission (test.csv)")
if st.button("Create submission.csv from dataset/test.csv"):
    if not model_loaded:
        st.error("Model not loaded.")
    elif not os.path.exists(TEST_CSV):
        st.error(f"Test CSV not found: {TEST_CSV}")
    else:
        with st.spinner("Preprocessing test.csv and generating submission..."):
            try:
                test_df = pd.read_csv(TEST_CSV).fillna("")
                if "combined_text" not in test_df.columns:
                    cat_series = safe_text_series(test_df, "catalog_content")
                    text_series = safe_text_series(test_df, "text")
                    test_df["combined_text"] = (cat_series + " " + text_series).str.lower()

                if not tf_loaded:
                    st.error("TF-IDF not loaded.")
                else:
                    X_text_sp = tf.transform(test_df["combined_text"].astype(str))
                    num_cols = ["ipq", "word_count", "char_count", "avg_word_len", "num_unique_words"]
                    if "word_count" not in test_df.columns:
                        stats = basic_text_stats(test_df["combined_text"])
                        for c in stats.columns:
                            test_df[c] = stats[c]
                    for c in num_cols:
                        if c not in test_df.columns:
                            test_df[c] = 0
                    X_num = csr_matrix(test_df[num_cols].astype(float).values)

                    use_image = False
                    if img_emb_loaded and img_df is not None and "sample_id" in img_df.columns:
                        try:
                            emb_dim = img_emb.shape[1] if img_emb.ndim == 2 else None
                        except Exception:
                            emb_dim = None
                        if emb_dim:
                            emb_map = dict(zip(img_df["sample_id"].astype(str), range(len(img_df))))
                            emb_rows = []
                            for sid in test_df["sample_id"].astype(str):
                                idx = emb_map.get(sid, None)
                                if idx is None:
                                    emb_rows.append(np.zeros(emb_dim, dtype=float))
                                else:
                                    emb_rows.append(img_emb[idx] if idx < len(img_emb) else np.zeros(emb_dim, dtype=float))
                            X_img = csr_matrix(np.vstack(emb_rows))
                            use_image = True

                    X_full = sparse_hstack([X_text_sp, X_num, X_img], format="csr") if use_image else sparse_hstack([X_text_sp, X_num], format="csr")
                    preds = None
                    try:
                        preds = model.predict(X_full.toarray() if hasattr(X_full, "toarray") and getattr(model, "__class__", None).__name__.startswith("CatBoost") else X_full)
                    except Exception as e:
                        st.error("Prediction failed (check shapes / feature order).")
                        st.error(str(e))
                        st.text(traceback.format_exc())

                    if preds is not None:
                        submission = pd.DataFrame({"sample_id": test_df["sample_id"], "price": preds})
                        os.makedirs(os.path.dirname(SUBMISSION_OUT) or ".", exist_ok=True)
                        submission.to_csv(SUBMISSION_OUT, index=False)
                        st.success(f"Submission saved → {SUBMISSION_OUT}")
                        st.dataframe(submission.head(10))

            except Exception as e:
                st.error(f"Unexpected error while preparing submission: {e}")
                st.text(traceback.format_exc())

# ---------------------------
# 7) Download artifacts
# ---------------------------
st.header("7) Download / Inspect artifacts")
col_a, col_b, col_c = st.columns(3)
with col_a:
    if os.path.exists(MODEL_PATH):
        try:
            st.download_button("Download model file", data=open(MODEL_PATH, "rb"), file_name=os.path.basename(MODEL_PATH))
        except Exception:
            pass
with col_b:
    if os.path.exists(TFIDF_PATH):
        try:
            st.download_button("Download TF-IDF (pkl)", data=open(TFIDF_PATH, "rb"), file_name=os.path.basename(TFIDF_PATH))
        except Exception:
            pass
with col_c:
    if os.path.exists(SUBMISSION_OUT):
        try:
            st.download_button("Download last submission", data=open(SUBMISSION_OUT, "rb"), file_name=os.path.basename(SUBMISSION_OUT))
        except Exception:
            pass

st.markdown("---")
st.write("Tips: If your TF-IDF vectorizer + model were trained with a different order of features than this app assumes, adapt the stacking order in training/inference to match your original training code.")
