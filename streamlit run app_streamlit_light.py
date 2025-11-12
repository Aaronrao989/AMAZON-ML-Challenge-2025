# app_streamlit_light.py
"""
Lightweight Streamlit app for safe inference (small memory footprint).
- Chunked prediction
- Streaming write to CSV
- Small-sample dry-run
- Avoids converting entire sparse feature matrix to dense
- Uses `width=` for images (no use_container_width warning)
"""

import os
import time
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st

from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.metrics import mean_squared_error

# ---------------------------
# Config
# ---------------------------
st.set_page_config(layout="wide", page_title="Smart Pricing — Light Inference")
DEFAULTS = {
    "DATA_PATH": "artifacts/cleaned_data.csv",
    "TFIDF_PATH": "artifacts/tfidf_vectorizer.pkl",
    "MODEL_PATH": "artifacts/model.pkl",
    "SUBMISSION_OUT": "artifacts/predictions_stream.csv",
    "IMG_NPY": None,
    "IMG_CSV": None,
    "IMAGES_DIR": "images"
}

DATA_PATH = st.sidebar.text_input("Cleaned data CSV", value=DEFAULTS["DATA_PATH"])
TFIDF_PATH = st.sidebar.text_input("TF-IDF vectorizer", value=DEFAULTS["TFIDF_PATH"])
MODEL_PATH = st.sidebar.text_input("Model path", value=DEFAULTS["MODEL_PATH"])
SUBMISSION_OUT = st.sidebar.text_input("Submission output", value=DEFAULTS["SUBMISSION_OUT"])
CHUNK_SIZE = st.sidebar.number_input("Prediction chunk size", min_value=16, max_value=4096, value=128, step=16)
DRY_RUN_ROWS = st.sidebar.number_input("Dry-run rows (small test)", min_value=4, max_value=1024, value=64, step=4)

uploaded_model = st.sidebar.file_uploader("Upload model (.pkl/.cbm)", type=["pkl", "cbm", "model"])

st.sidebar.markdown("Light mode: minimal UI, safe for low-memory hosts.")

# ---------------------------
# Helpers
# ---------------------------
def compute_rmse(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        mse = mean_squared_error(y_true, y_pred)
        return float(np.sqrt(mse))

def chunked_predict(model, X, chunk_size=128, catboost_dense=False):
    """
    Predict in chunks. Returns a numpy array or 2D array if model returns multi-output.
    If catboost_dense=True, convert each chunk to dense before predicting.
    """
    n = X.shape[0]
    preds = []
    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)
        Xc = X[s:e]
        # CatBoost often needs dense array; caller can set catboost_dense=True
        if catboost_dense:
            X_chunk_for_pred = Xc.toarray() if hasattr(Xc, "toarray") else Xc
        else:
            X_chunk_for_pred = Xc
        p = model.predict(X_chunk_for_pred)
        p = np.asarray(p)
        preds.append(p)
    if len(preds) == 0:
        return np.array([])
    # If 1D slices -> concatenate into 1D
    first = preds[0]
    if first.ndim == 1:
        return np.concatenate([p.reshape(-1) for p in preds], axis=0)
    else:
        return np.vstack(preds)

def save_predictions_stream(df_rows, preds, out_path, header_write=False):
    """
    Append predictions for df_rows (DataFrame) and preds (1D array) to CSV.
    header_write: if True, write headers (overwrites file).
    """
    df_out = pd.DataFrame({
        "sample_id": df_rows["sample_id"] if "sample_id" in df_rows.columns else np.arange(len(df_rows)),
        "predicted_price": preds
    })
    mode = "w" if header_write else "a"
    header = header_write
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df_out.to_csv(out_path, index=False, mode=mode, header=header)

def safe_load_joblib_or_catboost(path):
    """
    Try joblib.load, else try CatBoost native load_model.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    # try joblib
    try:
        return joblib.load(path)
    except Exception:
        pass
    # try catboost
    try:
        from catboost import CatBoostRegressor, CatBoostClassifier
        try:
            m = CatBoostRegressor()
            m.load_model(path)
            return m
        except Exception:
            m2 = CatBoostClassifier()
            m2.load_model(path)
            return m2
    except Exception:
        pass
    raise RuntimeError("Failed to load model (joblib or CatBoost).")

# ---------------------------
# Load artifacts
# ---------------------------
st.header("Light inference — load artifacts")

df = None
if os.path.exists(DATA_PATH):
    try:
        df = pd.read_csv(DATA_PATH)
        st.success(f"Loaded data: {DATA_PATH} (rows={len(df)})")
    except Exception as e:
        st.error(f"Failed to read data CSV: {e}")
else:
    st.info("No DATA_PATH found. Upload or set correct path in the sidebar.")

tf = None
if os.path.exists(TFIDF_PATH):
    try:
        tf = joblib.load(TFIDF_PATH)
        st.success("Loaded TF-IDF vectorizer")
    except Exception as e:
        st.error(f"Failed to load TF-IDF: {e}")
else:
    st.info("TF-IDF not found at TFIDF_PATH")

model = None
model_loaded = False
model_path_effective = MODEL_PATH
if uploaded_model is not None:
    tmp = Path("tmp_uploaded_model" + Path(uploaded_model.name).suffix)
    with open(tmp, "wb") as fh:
        fh.write(uploaded_model.read())
    model_path_effective = str(tmp)

if os.path.exists(model_path_effective):
    try:
        model = safe_load_joblib_or_catboost(model_path_effective)
        model_loaded = True
        st.success(f"Loaded model: {model_path_effective}")
    except Exception as e:
        st.error(f"Model load failed: {e}")
else:
    st.info("Model not found — upload or set MODEL_PATH")

# ---------------------------
# Minimal preview
# ---------------------------
if df is not None:
    st.subheader("Data preview")
    st.dataframe(df.head(50))

# ---------------------------
# Inference controls
# ---------------------------
st.header("Run inference (safe)")
if st.button("Run inference (stream to CSV)"):
    if not model_loaded:
        st.error("Model not loaded")
    elif df is None:
        st.error("No data loaded")
    elif tf is None:
        st.error("TF-IDF not loaded")
    else:
        try:
            # Build combined_text if not present
            df_local = df.fillna("")
            if "combined_text" not in df_local.columns:
                cat = df_local["catalog_content"] if "catalog_content" in df_local.columns else ""
                txt = df_local["text"] if "text" in df_local.columns else ""
                df_local["combined_text"] = (cat.astype(str) + " " + txt.astype(str)).str.lower()

            # Text -> sparse
            X_text_sp = tf.transform(df_local["combined_text"].astype(str))

            # numeric features (ensure minimal set present)
            num_cols = ["ipq", "word_count", "char_count", "avg_word_len", "num_unique_words"]
            if "word_count" not in df_local.columns:
                # compute light text stats
                words = df_local["combined_text"].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
                chars = df_local["combined_text"].str.len()
                uniq = df_local["combined_text"].str.split().apply(lambda toks: len(set(toks)) if isinstance(toks, list) else 0)
                avg_w = df_local["combined_text"].apply(lambda s: np.mean([len(w) for w in s.split()]) if len(s.split())>0 else 0)
                df_local["word_count"] = words
                df_local["char_count"] = chars
                df_local["num_unique_words"] = uniq
                df_local["avg_word_len"] = avg_w
            for c in num_cols:
                if c not in df_local.columns:
                    df_local[c] = 0
            X_num = csr_matrix(df_local[num_cols].astype(float).values)

            # stack features (no images in light mode)
            X_full = sparse_hstack([X_text_sp, X_num], format="csr")
            st.write("Built features shape:", X_full.shape)

            # If model expects dense for predict (CatBoost), we'll do per-chunk dense conversion.
            model_name = getattr(model, "__class__", type(model)).__name__ if model is not None else ""
            is_catboost = model_name.startswith("CatBoost")

            # Dry-run on small sample
            dry_n = min(int(DRY_RUN_ROWS), X_full.shape[0])
            st.info(f"Dry-run predicting {dry_n} rows to validate model compatibility")
            try:
                test_preds = chunked_predict(model, X_full[:dry_n], chunk_size=min(32, dry_n), catboost_dense=is_catboost)
                st.write("Dry-run preds shape:", np.asarray(test_preds).shape)
            except Exception as e:
                st.error(f"Dry-run predict failed: {e}")
                st.text(traceback.format_exc())
                raise

            # Stream predictions to CSV in chunks
            out_path = SUBMISSION_OUT
            # Remove existing file if present
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass

            total = X_full.shape[0]
            st.info(f"Starting chunked prediction: total rows={total}, chunk_size={CHUNK_SIZE}")
            written_header = False
            progress_bar = st.progress(0)
            for start in range(0, total, CHUNK_SIZE):
                end = min(total, start + CHUNK_SIZE)
                X_chunk = X_full[start:end]
                # convert chunk if needed
                try:
                    preds_chunk = chunked_predict(model, X_chunk, chunk_size=CHUNK_SIZE, catboost_dense=is_catboost)
                except Exception as e:
                    st.error(f"Prediction failed on chunk [{start}:{end}]: {e}")
                    st.text(traceback.format_exc())
                    raise

                preds_chunk = np.asarray(preds_chunk)
                # If preds 2D with single column -> flatten
                if preds_chunk.ndim == 2 and preds_chunk.shape[1] == 1:
                    preds_chunk = preds_chunk.reshape(-1)
                # If multi-output, error out (light app expects single-output regression)
                if preds_chunk.ndim == 2 and preds_chunk.shape[1] > 1:
                    st.error(f"Model returned multi-dimensional output with shape {preds_chunk.shape}. Aborting.")
                    raise RuntimeError("Multi-output predictions not supported in light mode")

                # Write chunk results
                if "sample_id" in df_local.columns:
                    rows_df = df_local.iloc[start:end][["sample_id"]].reset_index(drop=True)
                else:
                    rows_df = pd.DataFrame({"sample_id": np.arange(start, end)})

                header_write = not written_header
                save_predictions_stream(rows_df, preds_chunk, out_path, header_write=header_write)
                written_header = True

                progress_bar.progress(min(1.0, end / total))
                st.write(f"Wrote rows {start}:{end}")

            st.success(f"Streaming prediction finished → {out_path}")

            # Show few rows
            try:
                preview = pd.read_csv(out_path).head(20)
                st.dataframe(preview)
            except Exception:
                st.info("Could not preview output file.")

        except Exception as final_e:
            st.error(f"Fatal error during inference: {final_e}")
            st.text(traceback.format_exc())

# ---------------------------
# Tiny image preview (uses width= to avoid deprecation warning)
# ---------------------------
st.header("Images (optional)")
img_dir = Path(DEFAULTS["IMAGES_DIR"])
if img_dir.exists():
    files = sorted(img_dir.glob("*"))[:6]
    if files:
        cols = st.columns(min(3, len(files)))
        for i, f in enumerate(files):
            try:
                cols[i % len(cols)].image(str(f), caption=f.name, width=200)  # width param instead of use_container_width
            except Exception:
                cols[i % len(cols)].write("Failed to load image")
else:
    st.write("No images folder found (optional).")

st.markdown("---")
st.write("Notes: light mode streams predictions to disk chunk-by-chunk and avoids creating large dense arrays. Increase CHUNK_SIZE only if host has sufficient memory.")
