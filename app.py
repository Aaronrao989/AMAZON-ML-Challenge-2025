# app.py
"""
Streamlit dashboard for the Smart Product Pricing project.

Usage:
    .venv activated
    pip install streamlit pandas numpy joblib pillow matplotlib altair scipy catboost
    streamlit run app.py

This app assumes your project artifacts live in 'artifacts/' and model/vectorizer are saved there.
Paths are configurable via the sidebar.
"""

import os
import io
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# sparse helpers
from scipy.sparse import csr_matrix, hstack as sparse_hstack

st.set_page_config(layout="wide", page_title="Smart Pricing — Project Explorer")

# ---------------------------
# Sidebar: paths + quick config
# ---------------------------
st.sidebar.title("Config / Artifacts")
DATA_PATH = st.sidebar.text_input("Cleaned data CSV",
    value="/Users/aaronrao/Desktop/screenshots/student_resource/artifacts/cleaned_data.csv")
TEST_CSV = st.sidebar.text_input("Test CSV (raw)",
    value="/Users/aaronrao/Desktop/screenshots/student_resource/dataset/test.csv")
TFIDF_PATH = st.sidebar.text_input("TF-IDF vectorizer",
    value="/Users/aaronrao/Desktop/screenshots/student_resource/artifacts/tfidf_vectorizer.pkl")
MODEL_PATH = st.sidebar.text_input("CatBoost model",
    value="/Users/aaronrao/Desktop/screenshots/student_resource/artifacts/catboost_model.pkl")
IMG_NPY = st.sidebar.text_input("Image embeddings (.npy)",
    value="/Users/aaronrao/Desktop/screenshots/student_resource/artifacts/image_embeddings.npy")
IMG_CSV = st.sidebar.text_input("Image embeddings (.csv)",
    value="/Users/aaronrao/Desktop/screenshots/student_resource/artifacts/image_embeddings.csv")
IMAGES_DIR = st.sidebar.text_input("Local images folder",
    value="/Users/aaronrao/Desktop/screenshots/student_resource/images")
SUBMISSION_OUT = st.sidebar.text_input("Submission output",
    value="/Users/aaronrao/Desktop/screenshots/student_resource/artifacts/submission.csv")
MAX_IMAGE_PREVIEW = st.sidebar.slider("Max images to preview", 1, 16, 6)

st.sidebar.markdown("---")
st.sidebar.markdown("Built for: Smart Product Pricing (TF-IDF + Image embeddings + CatBoost)")

# Sidebar uploader fallback
st.sidebar.markdown("### Optional: Upload model file")
uploaded_model = st.sidebar.file_uploader("Upload model (.pkl / .cbm / .bin / .model)", type=["pkl", "cbm", "bin", "model"])

# ---------------------------
# Helpers (cached)
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
    """
    Robust model loader:
    - try joblib.load (pickles)
    - if that fails and catboost is available, try CatBoostRegressor().load_model()
    Raises on failure.
    """
    # try joblib first (covers sklearn-like pickles)
    try:
        return joblib.load(path)
    except Exception as e_joblib:
        last_err = e_joblib

    # try CatBoost native loader (lazy import)
    try:
        from catboost import CatBoostRegressor, CatBoostClassifier
        ext = Path(path).suffix.lower()
        # If extension looks like a catboost native model, load it
        if ext in {".cbm", ".bin", ".model"}:
            m = CatBoostRegressor()
            m.load_model(path)
            return m
        else:
            # Try generic CatBoost load
            try:
                m = CatBoostRegressor()
                m.load_model(path)
                return m
            except Exception as e_cb:
                last_err = e_cb
    except Exception as e_imp:
        last_err = e_imp

    raise RuntimeError(f"Failed to load model with joblib or CatBoost. Last error:\n{last_err}")

def basic_text_stats(s: pd.Series):
    """Return a dataframe of basic text stats for the series."""
    s = s.fillna("").astype(str)
    # n_words computation: split on whitespace
    n_words = s.str.split().apply(lambda ws: len(ws) if isinstance(ws, list) else 0)
    n_unique = s.str.split().apply(lambda ws: len(set(ws)) if isinstance(ws, list) else 0)
    return pd.DataFrame({
        "n_chars": s.str.len(),
        "n_words": n_words,
        "n_unique_words": n_unique
    })

def safe_text_series(df, col_name):
    """
    Return a Series of strings for column `col_name` in df.
    If the column exists, returns df[col_name].fillna("").astype(str)
    If it doesn't exist, returns an empty-string Series aligned with df.
    """
    if col_name in df.columns:
        return df[col_name].fillna("").astype(str)
    else:
        # create Series with same index so concatenation preserves alignment
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

# small utility to try to get TF-IDF attributes safely
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
# Main layout
# ---------------------------
st.title("Smart Product Pricing — Project Explorer")
st.markdown("Visualize dataset, features, images and run inference using your saved model.")

# 1) Dataset preview / upload
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
        # pick a sensible text column
        text_col = "combined_text" if "combined_text" in df.columns else ("catalog_content" if "catalog_content" in df.columns else (df.columns[0] if len(df.columns) > 0 else None))
        st.write("Using text column:", text_col)
        if text_col is not None:
            stats = basic_text_stats(df[text_col])
            st.write(stats.describe().T)
            # show histogram-ish using bar_chart of word counts (clipped)
            st.bar_chart(stats["n_words"].clip(0, 100).value_counts().sort_index().head(50))
        else:
            st.write("No columns available to compute text stats.")

# 2) TF-IDF details
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

# 3) Image embeddings preview
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
        # pick images corresponding to sample ids if available in img_df
        if img_df is not None and "sample_id" in img_df.columns:
            preview_ids = img_df["sample_id"].head(MAX_IMAGE_PREVIEW).astype(str).tolist()
            picked = []
            for sid in preview_ids:
                fname = os.path.join(IMAGES_DIR, f"img_{sid}.jpg")
                if os.path.exists(fname):
                    picked.append(fname)
            if not picked:
                # fallback: list first images found
                picked = [str(x) for x in list(Path(IMAGES_DIR).glob("*"))[:MAX_IMAGE_PREVIEW]]
            if picked:
                show_image_grid(picked, cols=3)
            else:
                st.write("No matching images found in images dir.")
        else:
            # just show first N images from folder
            files = sorted(Path(IMAGES_DIR).glob("*"))[:MAX_IMAGE_PREVIEW]
            if files:
                show_image_grid([str(x) for x in files], cols=3)
            else:
                st.write("No images present in folder.")
    else:
        st.write("Images dir not found:", IMAGES_DIR)

# ---------------------------
# 4) Model & Inference
# ---------------------------
st.header("4) Model & Inference")
model_loaded = False
model = None

# Decide model path source: uploaded or configured path
model_source_path = None
if uploaded_model is not None:
    # save upload to a temp file and use that path
    tmp_model_path = os.path.join(".", "tmp_uploaded_model" + Path(uploaded_model.name).suffix)
    with open(tmp_model_path, "wb") as f:
        f.write(uploaded_model.read())
    model_source_path = tmp_model_path
else:
    model_source_path = MODEL_PATH

if os.path.exists(model_source_path):
    try:
        model = load_model(model_source_path)
        model_loaded = True
        st.success(f"Loaded model from: {model_source_path}")
    except Exception as e:
        st.error("Could not load model. See details below.")
        st.text(str(e))
        st.text(traceback.format_exc())
        st.info("If this is a CatBoost model, ensure 'catboost' is installed: pip install catboost")
        st.info("If you trained with joblib/pickle, ensure the environment has the same packages available (e.g., catboost).")
else:
    st.info(f"Model not found at configured path: {model_source_path}")
    st.info("You can either update the MODEL_PATH in the sidebar or upload a model file (pkl/cbm) using the sidebar uploader.")

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
                # make combined_text only if missing - use safe_text_series helper
                if "combined_text" not in df_local.columns:
                    cat_series = safe_text_series(df_local, "catalog_content")
                    text_series = safe_text_series(df_local, "text")
                    df_local["combined_text"] = (cat_series + " " + text_series).str.lower()

                # safe TF-IDF transform (keep sparse)
                if not tf_loaded:
                    st.error("TF-IDF not loaded; cannot transform text.")
                else:
                    X_text_sp = tf.transform(df_local["combined_text"].astype(str))  # sparse matrix

                    # numeric features: ensure present and convert to sparse
                    num_cols = ["ipq", "word_count", "char_count", "avg_word_len", "num_unique_words"]
                    if "word_count" not in df_local.columns:
                        st.info("Computing text stats for numeric columns...")
                        stats = basic_text_stats(df_local["combined_text"])
                        for c in stats.columns:
                            df_local[c] = stats[c]

                    for c in num_cols:
                        if c not in df_local.columns:
                            df_local[c] = 0

                    X_num = csr_matrix(df_local[num_cols].astype(float).values)

                    # image embeddings (align) -> convert to sparse
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
                        else:
                            st.warning("Image embeddings could not determine embedding dim; skipping image features.")
                            use_image = False
                    else:
                        use_image = False

                    # stack features - always use sparse stacking to avoid dense blowup
                    if use_image:
                        X_full = sparse_hstack([X_text_sp, X_num, X_img], format="csr")
                    else:
                        X_full = sparse_hstack([X_text_sp, X_num], format="csr")

                    st.write("Feature matrix shape (sparse):", X_full.shape)

                    # predict (handle potential errors)
                    preds = None
                    try:
                        preds = model.predict(X_full)
                    except Exception as e:
                        st.error("Model prediction failed. This is often due to a shape/feature-order mismatch.")
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
# 5) Generate official submission (test.csv)
# ---------------------------
st.header("5) Generate submission (test.csv)")
if st.button("Create submission.csv from dataset/test.csv"):
    if not model_loaded:
        st.error("Model not loaded.")
    elif not os.path.exists(TEST_CSV):
        st.error(f"Test CSV not found: {TEST_CSV}")
    else:
        with st.spinner("Preprocessing test.csv and generating submission..."):
            try:
                test_df = pd.read_csv(TEST_CSV).fillna("")
                # build combined text if absent - use safe_text_series helper
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

                    # attach image embeddings if available
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
                        preds = model.predict(X_full)
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
# 6) Download artifacts
# ---------------------------
st.header("6) Download / Inspect artifacts")
col_a, col_b, col_c = st.columns(3)
with col_a:
    if os.path.exists(model_source_path):
        try:
            st.download_button("Download model file", data=open(model_source_path, "rb"), file_name=os.path.basename(model_source_path))
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
st.write("Tips: If your TF-IDF vectorizer + model were trained with a different order of features than this app assumes, you'll need to adapt the stacking order used in the inference section to match your training code (TF-IDF first, numeric features second, image embeddings last).")
