# app_multimodel.py
"""
Streamlit app for Smart Product Pricing — supports both text-only and multimodal (text+image) models.
Features:
 - pick model from artifacts or upload
 - loads TF-IDF vectorizer
 - loads image embeddings (.npy + optional .csv alignment)
 - builds features (TF-IDF sparse, numeric text stats, image embeddings)
 - shows diagnostics and feature-count checks
 - optional zero-padding to match model feature count (debugging only)
 - inference on artifacts/cleaned_data.csv and dataset/test.csv to produce submission.csv
"""

import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from PIL import Image

st.set_page_config(layout="wide", page_title="Smart Pricing — MultiModel Explorer")

# ---------- CONFIG defaults ----------
ROOT = Path("/Users/aaronrao/Desktop/screenshots/student_resource")
ARTIFACTS = ROOT / "artifacts"
DATA_CLEANED = ARTIFACTS / "cleaned_data.csv"
TEST_CSV = ROOT / "dataset" / "test.csv"
TFIDF_PATH_DEFAULT = ARTIFACTS / "tfidf_vectorizer.pkl"
IMG_NPY_DEFAULT = ARTIFACTS / "image_embeddings.npy"
IMG_CSV_DEFAULT = ARTIFACTS / "image_embeddings.csv"

# ---------- Helpers ----------
def list_model_candidates(artifacts_dir: Path):
    """Return candidate model files in artifacts dir."""
    if not artifacts_dir.exists():
        return []
    exts = (".pkl", ".cbm", ".bin", ".model", ".txt")
    files = [str(p) for p in artifacts_dir.iterdir() if p.suffix.lower() in exts]
    return sorted(files)

def safe_load_joblib(path):
    return joblib.load(path)

def safe_load_tfidf(path):
    v = joblib.load(path)
    # try to get vocab size gracefully
    try:
        vocab_size = len(v.get_feature_names_out())
    except Exception:
        try:
            vocab_size = len(v.get_feature_names())
        except Exception:
            vocab_size = None
    return v, vocab_size

def safe_load_npy(path):
    return np.load(path)

def basic_text_features_series(s: pd.Series):
    s = s.fillna("").astype(str)
    words = s.str.split()
    n_words = words.apply(lambda ws: len(ws) if isinstance(ws, list) else 0)
    n_chars = s.str.len()
    n_unique = words.apply(lambda ws: len(set(ws)) if isinstance(ws, list) else 0)
    avg_word_len = words.apply(lambda ws: np.mean([len(w) for w in ws]) if isinstance(ws, list) and len(ws)>0 else 0.0)
    return pd.DataFrame({
        "word_count": n_words,
        "char_count": n_chars,
        "avg_word_len": avg_word_len,
        "num_unique_words": n_unique
    })

def detect_model_type_and_expected(model_obj, model_path):
    """
    Return (backend, expected_feature_count, feature_names_or_None)
    backend: 'lightgbm', 'catboost', 'sklearn_joblib', 'unknown'
    """
    # LightGBM Booster
    try:
        import lightgbm as lgb
        if isinstance(model_obj, lgb.Booster):
            try:
                return "lightgbm_booster", model_obj.num_feature(), model_obj.feature_name()
            except Exception:
                return "lightgbm_booster", None, None
        # sklearn LGBMRegressor loaded via joblib
        from lightgbm import LGBMRegressor
        if isinstance(model_obj, LGBMRegressor):
            if hasattr(model_obj, "booster_") and model_obj.booster_ is not None:
                try:
                    b = model_obj.booster_
                    return "lightgbm_sklearn", b.num_feature(), b.feature_name()
                except Exception:
                    pass
            if hasattr(model_obj, "n_features_in_"):
                return "lightgbm_sklearn", int(model_obj.n_features_in_), getattr(model_obj, "feature_names_in_", None)
    except Exception:
        pass

    # CatBoost
    try:
        from catboost import CatBoostRegressor
        if isinstance(model_obj, CatBoostRegressor):
            try:
                cnt = model_obj.get_feature_count()
                names = getattr(model_obj, "feature_names_", None)
                return "catboost", cnt, names
            except Exception:
                return "catboost", None, None
    except Exception:
        pass

    # sklearn-like (joblib)
    try:
        if hasattr(model_obj, "n_features_in_"):
            return "sklearn_joblib", int(model_obj.n_features_in_), getattr(model_obj, "feature_names_in_", None)
    except Exception:
        pass

    return "unknown", None, None

def load_model_robust(path: str):
    """
    Try joblib.load, else try LightGBM booster load, else try CatBoost native loading.
    Returns the loaded object or raises.
    """
    last_err = None
    # joblib first
    try:
        m = joblib.load(path)
        return m
    except Exception as e:
        last_err = e
    # try LightGBM Booster (model_file)
    try:
        import lightgbm as lgb
        b = lgb.Booster(model_file=path)
        return b
    except Exception as e:
        last_err = e
    # try CatBoost native
    try:
        from catboost import CatBoostRegressor
        m = CatBoostRegressor()
        m.load_model(path)
        return m
    except Exception as e:
        last_err = e
    raise RuntimeError(f"Failed to load model '{path}'. Last error: {last_err}")

def pad_or_trim_sparse(X, target_cols):
    """If X has fewer columns than target, pad zeros. If more, trim to target_cols (risky)."""
    n_rows, n_cols = X.shape
    if target_cols is None:
        return X, 0
    if n_cols == target_cols:
        return X, 0
    if n_cols < target_cols:
        pad = csr_matrix((n_rows, target_cols - n_cols), dtype=X.dtype)
        X2 = sparse_hstack([X, pad], format="csr")
        return X2, target_cols - n_cols
    # trim (dangerous) — will simply slice columns
    X2 = X[:, :target_cols]
    return X2, n_cols - target_cols

def predict_with_model(model_obj, X):
    """Call model.predict with appropriate conversion if needed."""
    # LightGBM Booster
    try:
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
        if isinstance(model_obj, lgb.Booster):
            # Booster expects 2d np array or csr_matrix; pass as is
            return model_obj.predict(X)
        if isinstance(model_obj, LGBMRegressor):
            return model_obj.predict(X)
    except Exception:
        pass
    # CatBoost
    try:
        from catboost import CatBoostRegressor
        if isinstance(model_obj, CatBoostRegressor):
            # CatBoost prefers dense numpy arrays; convert
            if hasattr(X, "toarray"):
                Xd = X.toarray()
            else:
                Xd = np.asarray(X)
            return model_obj.predict(Xd)
    except Exception:
        pass
    # sklearn/pipeline-like
    try:
        if hasattr(model_obj, "predict"):
            # Some sklearn models accept sparse, some not; convert to dense if necessary
            try:
                return model_obj.predict(X)
            except Exception:
                if hasattr(X, "toarray"):
                    return model_obj.predict(X.toarray())
                else:
                    return model_obj.predict(np.asarray(X))
    except Exception as e:
        raise
    raise RuntimeError("Cannot predict with unknown model type")

# ---------- UI ----------
st.title("Smart Pricing — multimodel explorer")

# left sidebar controls
st.sidebar.header("Paths & artifacts")
MODEL_PATH = st.sidebar.text_input("Model path (or leave blank to pick below)", value=str(ARTIFACTS / "catboost_model.pkl"))
tfidf_path_input = st.sidebar.text_input("TF-IDF vectorizer", value=str(TFIDF_PATH_DEFAULT))
img_npy_input = st.sidebar.text_input("Image embeddings (.npy)", value=str(IMG_NPY_DEFAULT))
img_csv_input = st.sidebar.text_input("Image embeddings (.csv)", value=str(IMG_CSV_DEFAULT))
st.sidebar.markdown("---")

candidates = list_model_candidates(ARTIFACTS)
st.sidebar.write("Models found in artifacts:")
for c in candidates:
    st.sidebar.write("•", c)
st.sidebar.markdown("---")
uploaded_model = st.sidebar.file_uploader("Upload model file (optional)", type=["pkl", "cbm", "bin", "model", "txt"])
use_padding = st.sidebar.checkbox("Enable zero-pad/truncate to match model features (debug only)", value=False)
st.sidebar.caption("Padding is risky. Prefer using the same TF-IDF and model used during training.")

# main: load TF-IDF and images
st.header("1) Artifacts & diagnostics")
tfidf = None
tf_vocab = None
if tfidf_path_input and os.path.exists(tfidf_path_input):
    try:
        tfidf, tf_vocab = safe_load_tfidf(tfidf_path_input)
        st.success(f"Loaded TF-IDF vectorizer, vocab size ~ {tf_vocab}")
    except Exception as e:
        st.error(f"Failed to load TF-IDF vectorizer: {e}")
else:
    st.info("TF-IDF vectorizer not found at path. Set path in sidebar or upload vectorizer.")

img_emb = None
img_df = None
if img_npy_input and os.path.exists(img_npy_input):
    try:
        img_emb = safe_load_npy(img_npy_input)
        st.success(f"Loaded image embeddings (.npy) shape: {getattr(img_emb, 'shape', None)}")
    except Exception as e:
        st.error(f"Failed to load image .npy: {e}")
if img_csv_input and os.path.exists(img_csv_input):
    try:
        img_df = pd.read_csv(img_csv_input)
        st.success("Loaded image embeddings CSV for alignment (contains sample_id?)")
    except Exception as e:
        st.error(f"Failed to load image embeddings CSV: {e}")

# model selection / upload
st.header("2) Choose model")
model_path_to_use = None
if uploaded_model is not None:
    tmp = ARTIFACTS / f"uploaded_{uploaded_model.name}"
    with open(tmp, "wb") as f:
        f.write(uploaded_model.read())
    model_path_to_use = str(tmp)
    st.info(f"Using uploaded model: {tmp}")
else:
    # if user provided a path in sidebar, use that if exists; else show found candidates dropdown
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        model_path_to_use = MODEL_PATH
    elif candidates:
        model_choice = st.selectbox("Choose model from artifacts", candidates, index=0)
        model_path_to_use = model_choice
    else:
        st.warning("No model path available. Provide path in sidebar or upload a model file.")
        model_path_to_use = None

st.write("Model path in use:", model_path_to_use)

model_obj = None
model_load_err = None
if model_path_to_use:
    try:
        model_obj = load_model_robust(model_path_to_use)
        st.success("Model loaded.")
    except Exception as e:
        model_load_err = str(e)
        st.error("Failed to load model: " + model_load_err)
        st.text(traceback.format_exc())

# show model expected features
expected_count = None
feat_names = None
backend = None
if model_obj is not None:
    backend, expected_count, feat_names = detect_model_type_and_expected(model_obj, model_path_to_use)
    st.write("Model backend:", backend)
    st.write("Model expected feature count:", expected_count)
    if feat_names is not None:
        st.write("Sample feature names (first 20):", feat_names[:20])

# small preview of cleaned_data
st.header("3) Preview cleaned_data (as loaded by app)")
df = None
if DATA_CLEANED.exists():
    try:
        df = pd.read_csv(DATA_CLEANED)
        st.write(f"cleaned_data.csv rows: {len(df)}, columns: {list(df.columns)[:20]}")
        st.dataframe(df.head(5))
    except Exception as e:
        st.error("Failed to load cleaned_data.csv: " + str(e))
else:
    st.info("No artifacts/cleaned_data.csv found at default path. You may upload or set a different path.")

# ---------- feature builder UI ----------
st.header("4) Build features and run quick check")
build_btn = st.button("Build features for cleaned_data and check shape")
if build_btn:
    if df is None:
        st.error("No cleaned_data.csv loaded to build features.")
    elif tfidf is None:
        st.error("No TF-IDF vectorizer loaded — cannot build text features.")
    else:
        try:
            df_local = df.copy().fillna("")
            # combined_text
            if "combined_text" not in df_local.columns:
                df_local["combined_text"] = (df_local.get("catalog_content", "").astype(str) + " " + df_local.get("text", "").astype(str)).str.lower()

            # TF-IDF sparse
            X_text_sp = tfidf.transform(df_local["combined_text"].astype(str))
            st.write("TF-IDF sparse shape:", X_text_sp.shape, "vocab size:", tf_vocab)

            # numeric text stats
            stats = basic_text_features_series(df_local["combined_text"])
            for c in stats.columns:
                df_local[c] = stats[c]

            num_cols = ["ipq", "word_count", "char_count", "avg_word_len", "num_unique_words"]
            for col in num_cols:
                if col not in df_local.columns:
                    df_local[col] = 0
            X_num = csr_matrix(df_local[num_cols].astype(float).values)
            st.write("Numeric features shape:", X_num.shape)

            # image embedding alignment (if available)
            use_img = False
            X_img = None
            if img_emb is not None and img_df is not None and "sample_id" in img_df.columns:
                emb_dim = img_emb.shape[1] if img_emb.ndim == 2 else None
                if emb_dim:
                    map_idx = dict(zip(img_df["sample_id"].astype(str), range(len(img_df))))
                    rows = []
                    for sid in df_local["sample_id"].astype(str):
                        idx = map_idx.get(sid, None)
                        if idx is None:
                            rows.append(np.zeros(emb_dim, dtype=float))
                        else:
                            rows.append(img_emb[idx] if idx < len(img_emb) else np.zeros(emb_dim, dtype=float))
                    X_img = csr_matrix(np.vstack(rows))
                    use_img = True
                    st.write("Image features shape:", X_img.shape)
            else:
                # if no CSV alignment but .npy is present, attempt to use same ordering if shapes match
                if img_emb is not None and img_emb.shape[0] == len(df_local):
                    X_img = csr_matrix(img_emb)
                    use_img = True
                    st.write("Image features shape (direct npy match):", X_img.shape)

            # stack features
            mats = [X_text_sp, X_num]
            if use_img:
                mats.append(X_img)
            X_full = sparse_hstack(mats, format="csr")
            st.write("Built full feature matrix shape:", X_full.shape)

            # show mismatch info vs model expected
            if model_obj is not None:
                st.write("Model expected features:", expected_count)
                if expected_count is not None and X_full.shape[1] != expected_count:
                    st.error(f"Feature count mismatch: built={X_full.shape[1]}  model_expected={expected_count}")
                    if use_padding:
                        X_full, delta = pad_or_trim_sparse(X_full, expected_count)
                        if delta >= 0:
                            st.warning(f"Padded {delta} zero columns to match model expected features. New shape: {X_full.shape}")
                        else:
                            st.warning(f"Trimmed {-delta} columns to match model expected features. New shape: {X_full.shape}")
                else:
                    st.success("Feature count matches model expected features (or model expected unknown).")

            # quick predict test (only if model loaded)
            if model_obj is not None:
                try:
                    preds = predict_with_model(model_obj, X_full)
                    st.write("Sample predictions (first 5):", preds[:5])
                except Exception as e:
                    st.error("Model predict failed: " + str(e))
                    st.text(traceback.format_exc())

        except Exception as be:
            st.error("Failed building features: " + str(be))
            st.text(traceback.format_exc())

# ---------- generate submission ----------
st.header("5) Generate submission.csv (from dataset/test.csv)")
if st.button("Create submission.csv from dataset/test.csv"):
    if model_obj is None:
        st.error("Load a model first.")
    elif tfidf is None:
        st.error("Load TF-IDF vectorizer first.")
    elif not os.path.exists(TEST_CSV):
        st.error(f"Test CSV not found at: {TEST_CSV}")
    else:
        try:
            test_df = pd.read_csv(TEST_CSV).fillna("")
            if "combined_text" not in test_df.columns:
                test_df["combined_text"] = (test_df.get("catalog_content", "").astype(str) + " " + test_df.get("text", "").astype(str)).str.lower()

            X_text_sp = tfidf.transform(test_df["combined_text"].astype(str))

            stats = basic_text_features_series(test_df["combined_text"])
            for c in stats.columns:
                test_df[c] = stats[c]

            num_cols = ["ipq", "word_count", "char_count", "avg_word_len", "num_unique_words"]
            for col in num_cols:
                if col not in test_df.columns:
                    test_df[col] = 0
            X_num = csr_matrix(test_df[num_cols].astype(float).values)

            # image alignment for test set
            use_img = False
            X_img = None
            if img_emb is not None and img_df is not None and "sample_id" in img_df.columns:
                emb_dim = img_emb.shape[1] if img_emb.ndim == 2 else None
                if emb_dim:
                    map_idx = dict(zip(img_df["sample_id"].astype(str), range(len(img_df))))
                    rows = []
                    for sid in test_df["sample_id"].astype(str):
                        idx = map_idx.get(sid, None)
                        if idx is None:
                            rows.append(np.zeros(emb_dim, dtype=float))
                        else:
                            rows.append(img_emb[idx] if idx < len(img_emb) else np.zeros(emb_dim, dtype=float))
                    X_img = csr_matrix(np.vstack(rows))
                    use_img = True
            else:
                if img_emb is not None and img_emb.shape[0] == len(test_df):
                    X_img = csr_matrix(img_emb)
                    use_img = True

            mats = [X_text_sp, X_num]
            if use_img:
                mats.append(X_img)
            X_full = sparse_hstack(mats, format="csr")

            # pad or trim if requested
            if expected_count is not None and X_full.shape[1] != expected_count:
                st.warning(f"Feature mismatch test: built={X_full.shape[1]} expected={expected_count}")
                if use_padding:
                    X_full, delta = pad_or_trim_sparse(X_full, expected_count)
                    st.warning(f"Applied pad/trim delta: {delta}; new shape: {X_full.shape}")
                else:
                    st.error("Enable padding toggle in sidebar to auto-fix (not recommended for final submit).")
                    st.stop()

            # predict
            preds = predict_with_model(model_obj, X_full)
            submission = pd.DataFrame({
                "sample_id": test_df["sample_id"],
                "price": preds
            })
            out = ARTIFACTS / "submission.csv"
            submission.to_csv(out, index=False)
            st.success(f"Saved submission → {out}")
            st.dataframe(submission.head(10))

        except Exception as e:
            st.error("Failed to create submission: " + str(e))
            st.text(traceback.format_exc())

# ---------- small utilities ----------
st.header("6) Utilities")
if st.button("Show TF-IDF sample tokens (first 50)"):
    if tfidf is None:
        st.error("TF-IDF not loaded.")
    else:
        try:
            try:
                toks = list(tfidf.get_feature_names_out())[:50]
            except Exception:
                toks = list(tfidf.get_feature_names())[:50]
            st.write(toks)
        except Exception as e:
            st.error("Failed to list tokens: " + str(e))

st.markdown("---")
st.write("Notes:")
st.write("- Use the exact TF-IDF vectorizer that was used during training to avoid feature-count mismatches.")
st.write("- Padding/trimming is only for debugging and may produce invalid predictions; retrain the model with your desired TF-IDF dimension or point the app to the original vectorizer for production.")
