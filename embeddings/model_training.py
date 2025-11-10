# import os
# import sys
# import joblib
# import numpy as np
# import pandas as pd
# from scipy.sparse import hstack, csr_matrix
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import KFold
# import lightgbm as lgb
# from tqdm import tqdm

# # -----------------------
# # CONFIG
# # -----------------------
# CLEANED_PATH = "artifacts/cleaned_data.csv"         # produced by data_preprocessing.py
# IMAGE_EMB_PATH = "artifacts/image_embeddings.npy"  # optional, produced by image_features.py
# ARTIFACT_DIR = "artifacts"
# TFIDF_PATH = os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl")
# TEXT_MODEL_PATH = os.path.join(ARTIFACT_DIR, "lgbm_text_model.pkl")
# MULTI_MODEL_PATH = os.path.join(ARTIFACT_DIR, "lgbm_multi_model.pkl")

# NUM_FOLDS = 5
# RANDOM_STATE = 42
# TFIDF_MAX_FEATURES = 10000

# # LightGBM params
# LGB_PARAMS = {
#     "objective": "regression",
#     "metric": "mae",
#     "learning_rate": 0.05,
#     "num_leaves": 31,
#     "feature_fraction": 0.9,
#     "bagging_fraction": 0.8,
#     "bagging_freq": 5,
#     "verbosity": -1,
#     "seed": RANDOM_STATE,
# }

# # -----------------------
# # METRICS
# # -----------------------
# def smape(y_true, y_pred, eps=1e-8):
#     num = np.abs(y_pred - y_true)
#     denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
#     denom = np.where(denom == 0, eps, denom)
#     return 100.0 * np.mean(num / denom)

# # -----------------------
# # UTILITIES
# # -----------------------
# def ensure_artifact_dir():
#     os.makedirs(ARTIFACT_DIR, exist_ok=True)

# def load_cleaned_data(path=CLEANED_PATH):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Cleaned data not found at: {path}")
#     df = pd.read_csv(path)
#     # Expect columns: at least sample_id, text, target
#     if "text" not in df.columns:
#         raise ValueError("Column 'text' missing in cleaned_data.csv")
#     if "target" not in df.columns:
#         raise ValueError("Column 'target' missing in cleaned_data.csv (should be log1p(price) or price).")
#     # If target looks like raw price (large variance), we'll convert inside train function
#     return df

# def load_image_embeddings(path=IMAGE_EMB_PATH):
#     if not os.path.exists(path):
#         print("ℹ️ No image embeddings found at artifacts. Continuing with text-only training.")
#         return None
#     emb = np.load(path)
#     print(f"✅ Loaded image embeddings shape: {emb.shape}")
#     return emb

# # -----------------------
# # FEATURE PREP
# # -----------------------
# def build_tfidf(train_texts, max_features=TFIDF_MAX_FEATURES):
#     print("🔤 Fitting TF-IDF vectorizer...")
#     vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), stop_words="english")
#     X_text = vec.fit_transform(train_texts)
#     return X_text, vec

# def transform_tfidf(vec, texts):
#     return vec.transform(texts)

# def combine_text_and_image(X_text_sparse, ipq_array, image_emb):
#     """
#     Combine sparse TF-IDF (csr_matrix) + ipq (n,1) + image embeddings (n, d)
#     Returns a sparse/dense combination suitable for LightGBM (it accepts scipy sparse)
#     """
#     # ipq as sparse column
#     ipq_col = csr_matrix(ipq_array.reshape(-1,1))
#     # image_emb might be dense numpy; convert to csr_matrix
#     image_sparse = csr_matrix(image_emb)
#     combined = hstack([X_text_sparse, ipq_col, image_sparse], format='csr')
#     return combined

# # -----------------------
# # TRAINING LOOPS
# # -----------------------
# def train_lgb_cv(X, y, description="Model", params=LGB_PARAMS, n_splits=NUM_FOLDS):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
#     oof_preds = np.zeros(len(y))
#     fold_smape = []
#     print(f"🚀 Starting {n_splits}-Fold CV for {description} ...")
#     # tqdm over folds
#     for fold, (tr_idx, val_idx) in enumerate(tqdm(list(kf.split(X)), desc=f"{description} folds", total=n_splits)):
#         print(f"\n📂 {description} - Fold {fold+1}/{n_splits}")
#         X_tr = X[tr_idx]
#         X_val = X[val_idx]
#         y_tr = y[tr_idx]
#         y_val = y[val_idx]

#         dtrain = lgb.Dataset(X_tr, label=y_tr)
#         dval = lgb.Dataset(X_val, label=y_val)

#         model = lgb.train(
#             params,
#             dtrain,
#             valid_sets=[dtrain, dval],
#             num_boost_round=2000,
#             callbacks=[
#                 lgb.early_stopping(stopping_rounds=50),
#                 lgb.log_evaluation(period=200)
#             ],
#         )

#         val_pred = model.predict(X_val, num_iteration=model.best_iteration)
#         # val_pred and y_val are on log-scale (if we trained on log). The caller must decide.
#         oof_preds[val_idx] = val_pred

#         fold_smape_val = None  # caller will compute using inverse transform
#         # Save last model of final fold later or return model from last fold
#     return oof_preds, model  # return last trained model too

# # -----------------------
# # MAIN
# # -----------------------
# def main():
#     ensure_artifact_dir()

#     # 1) Load cleaned data
#     df = load_cleaned_data(CLEANED_PATH)
#     print(f"✅ Loaded {len(df)} samples from cleaned_data.csv")

#     # Detect if target is log1p already (heuristic: if median < 10 and max small -> assume log)
#     # We'll support both: if 'is_log' flag isn't present, assume cleaned_data target is log1p(price).
#     # To be safe, if median(target) > 100, assume raw price and convert to log.
#     is_log_target = False
#     if df["target"].median() < 20 and df["target"].max() < 100:  # heuristic
#         # it's probably log1p already (since your earlier pipeline created log_price)
#         is_log_target = True
#     else:
#         # if target seems large -> assume raw price, convert to log
#         print("ℹ️ 'target' column appears raw. Converting to log1p for stable training.")
#         df["target"] = np.log1p(df["target"])
#         is_log_target = True

#     # 2) Fit TF-IDF on full training text
#     X_text_full, tfidf = build_tfidf(df["text"].fillna("").astype(str))
#     joblib.dump(tfidf, TFIDF_PATH)
#     print(f"✅ Saved TF-IDF vectorizer to {TFIDF_PATH}")

#     y = df["target"].values  # log(target)

#     # -------------------------
#     # TEXT-ONLY TRAINING (full dataset)
#     # -------------------------
#     print("\n===== TEXT-ONLY TRAINING (full dataset) =====")
#     oof_text_preds, text_model = train_lgb_cv(X_text_full, y, description="Text-only")
#     # Inverse transform predictions to original price scale
#     oof_text_preds_exp = np.expm1(oof_text_preds)
#     # original prices: if original cleaned file had 'price' column, use that for SMAPE; else user must know
#     if "price" in df.columns:
#         price_true = df["price"].values
#         text_smape = smape(price_true, oof_text_preds_exp)
#         print(f"\n📊 Text-only OOF SMAPE: {text_smape:.4f}%")
#     else:
#         print("⚠️ 'price' column not present in cleaned_data.csv — cannot compute SMAPE on original scale.")

#     # Save text-only model
#     joblib.dump(text_model, TEXT_MODEL_PATH)
#     print(f"✅ Saved text-only LightGBM model to {TEXT_MODEL_PATH}")

#     # -------------------------
#     # MULTIMODAL TRAINING (if image embeddings exist)
#     # -------------------------
#     emb = load_image_embeddings(IMAGE_EMB_PATH)
#     if emb is None:
#         print("ℹ️ Skipping multimodal training (no embeddings).")
#         return

#     # Align embeddings to top rows of df (explicit assumption; warn if sizes mismatch)
#     n_emb = emb.shape[0]
#     n_df = len(df)
#     n_use = min(n_emb, n_df)
#     if n_emb != n_df:
#         print(f"⚠️ Number of embeddings ({n_emb}) != number of rows in cleaned_data ({n_df}).")
#         print(f"   → Will train multimodal model on first {n_use} rows of cleaned_data (index 0..{n_use-1}).")
#     else:
#         print("✅ Embeddings count matches cleaned_data rows; using all rows.")

#     df_sub = df.iloc[:n_use].reset_index(drop=True)
#     X_text_sub = X_text_full[:n_use]

#     # Extract ipq if present, else zeros
#     if "ipq" in df_sub.columns:
#         ipq = df_sub["ipq"].fillna(1).astype(float).values
#     else:
#         ipq = np.ones(n_use, dtype=float)

#     # combine features
#     print("🔗 Combining text TF-IDF + IPQ + image embeddings into a single feature matrix...")
#     X_multi = combine_text_and_image(X_text_sub, ipq, emb[:n_use])

#     y_sub = df_sub["target"].values

#     # Train CV on multimodal data
#     oof_multi_preds, multi_model = train_lgb_cv(X_multi, y_sub, description="Multimodal")
#     oof_multi_preds_exp = np.expm1(oof_multi_preds)

#     if "price" in df_sub.columns:
#         price_true_sub = df_sub["price"].values
#         multi_smape = smape(price_true_sub, oof_multi_preds_exp)
#         print(f"\n📊 Multimodal OOF SMAPE (on subset): {multi_smape:.4f}%")
#     else:
#         print("⚠️ 'price' column not present in cleaned_data.csv — cannot compute SMAPE on original scale for multimodal.")

#     # Save multimodal model
#     joblib.dump(multi_model, MULTI_MODEL_PATH)
#     print(f"✅ Saved multimodal LightGBM model to {MULTI_MODEL_PATH}")

# if __name__ == "__main__":
#     main()

#1
# import os
# import joblib
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from scipy.sparse import hstack, csr_matrix
# from sklearn.model_selection import KFold
# from sklearn.feature_extraction.text import TfidfVectorizer
# from catboost import CatBoostRegressor, Pool

# # Optional sentence transformer (semantic embeddings)
# try:
#     from sentence_transformers import SentenceTransformer
#     USE_SENTENCE_EMB = False
# except ImportError:
#     print("⚠️ sentence-transformers not installed — proceeding without semantic embeddings.")
#     USE_SENTENCE_EMB = False


# # ====================================================
# # CONFIG
# # ====================================================
# CLEANED_PATH = "artifacts/cleaned_data.csv"
# IMAGE_EMB_PATH = "artifacts/image_embeddings.npy"
# ARTIFACT_DIR = "artifacts"

# TFIDF_PATH = os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl")
# TEXT_MODEL_PATH = os.path.join(ARTIFACT_DIR, "catboost_text_model.pkl")
# MULTI_MODEL_PATH = os.path.join(ARTIFACT_DIR, "catboost_multi_model.pkl")

# NUM_FOLDS = 5
# RANDOM_STATE = 42
# TFIDF_MAX_FEATURES = 10000

# # CatBoost params
# CAT_PARAMS = {
#     "iterations": 3000,
#     "learning_rate": 0.03,
#     "depth": 8,
#     "loss_function": "MAE",
#     "eval_metric": "MAE",
#     "early_stopping_rounds": 100,
#     "random_seed": RANDOM_STATE,
#     "verbose": 200,
#     "task_type": "CPU",
# }


# # ====================================================
# # METRICS
# # ====================================================
# def smape(y_true, y_pred, eps=1e-8):
#     num = np.abs(y_pred - y_true)
#     denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
#     denom = np.where(denom == 0, eps, denom)
#     return 100.0 * np.mean(num / denom)


# # ====================================================
# # UTILITIES
# # ====================================================
# def ensure_artifact_dir():
#     os.makedirs(ARTIFACT_DIR, exist_ok=True)


# def load_cleaned_data(path=CLEANED_PATH):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Cleaned data not found at: {path}")
#     df = pd.read_csv(path)
#     if "text" not in df.columns:
#         raise ValueError("Column 'text' missing in cleaned_data.csv")
#     if "target" not in df.columns:
#         raise ValueError("Column 'target' missing in cleaned_data.csv (should be log1p(price) or price).")
#     return df


# def load_image_embeddings(path=IMAGE_EMB_PATH):
#     if not os.path.exists(path):
#         print("ℹ️ No image embeddings found. Continuing text-only training.")
#         return None
#     emb = np.load(path)
#     print(f"✅ Loaded image embeddings shape: {emb.shape}")
#     return emb


# # ====================================================
# # FEATURE PREPARATION
# # ====================================================
# def build_tfidf(train_texts, max_features=TFIDF_MAX_FEATURES):
#     print("🔤 Fitting TF-IDF vectorizer...")
#     vec = TfidfVectorizer(
#         max_features=max_features, ngram_range=(1, 2), stop_words="english"
#     )
#     X_text = vec.fit_transform(train_texts)
#     return X_text, vec


# def transform_tfidf(vec, texts):
#     return vec.transform(texts)


# def compute_sentence_embeddings(texts):
#     """Generate dense semantic embeddings using sentence-transformers"""
#     if not USE_SENTENCE_EMB:
#         return None
#     print("🧠 Generating sentence-transformer embeddings...")
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     emb = model.encode(texts, batch_size=256, show_progress_bar=True)
#     print(f"✅ Sentence embeddings shape: {emb.shape}")
#     return emb


# def combine_features(X_text_sparse, ipq_array, image_emb=None, sent_emb=None):
#     """
#     Combine all available features into one sparse/dense matrix
#     """
#     ipq_col = csr_matrix(ipq_array.reshape(-1, 1))
#     mats = [X_text_sparse, ipq_col]

#     if sent_emb is not None:
#         mats.append(csr_matrix(sent_emb))
#     if image_emb is not None:
#         mats.append(csr_matrix(image_emb))

#     combined = hstack(mats, format="csr")
#     return combined


# # ====================================================
# # TRAINING LOOP (CatBoost)
# # ====================================================
# def train_catboost_cv(X, y, description="Model", params=CAT_PARAMS, n_splits=NUM_FOLDS):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
#     oof_preds = np.zeros(len(y))
#     fold_scores = []

#     print(f"🚀 Starting {n_splits}-Fold CV for {description} ...")

#     for fold, (tr_idx, val_idx) in enumerate(tqdm(kf.split(X), total=n_splits, desc=f"{description} folds")):
#         print(f"\n📂 {description} - Fold {fold+1}/{n_splits}")

#         X_tr, X_val = X[tr_idx], X[val_idx]
#         y_tr, y_val = y[tr_idx], y[val_idx]

#         train_pool = Pool(X_tr, label=y_tr)
#         val_pool = Pool(X_val, label=y_val)

#         model = CatBoostRegressor(**params)
#         model.fit(train_pool, eval_set=val_pool, use_best_model=True)

#         val_pred = model.predict(X_val)
#         oof_preds[val_idx] = val_pred

#         val_pred_exp = np.expm1(val_pred)
#         y_val_exp = np.expm1(y_val)
#         fold_smape = smape(y_val_exp, val_pred_exp)
#         fold_scores.append(fold_smape)
#         print(f"📊 Fold {fold+1} SMAPE: {fold_smape:.4f}%")

#     print(f"\n✅ {description} Mean CV SMAPE: {np.mean(fold_scores):.4f}%")
#     return oof_preds, model


# # ====================================================
# # MAIN
# # ====================================================
# def main():
#     ensure_artifact_dir()

#     # 1️⃣ Load data
#     df = load_cleaned_data(CLEANED_PATH)
#     print(f"✅ Loaded {len(df)} samples from cleaned_data.csv")

#     # 2️⃣ Handle target scaling
#     is_log_target = False
#     if df["target"].median() < 20 and df["target"].max() < 100:
#         is_log_target = True
#     else:
#         print("ℹ️ 'target' appears raw. Applying log1p for stable training.")
#         df["target"] = np.log1p(df["target"])
#         is_log_target = True

#     y = df["target"].values
#     texts = df["text"].fillna("").astype(str)

#     # 3️⃣ Build TF-IDF features
#     X_tfidf, tfidf = build_tfidf(texts)
#     joblib.dump(tfidf, TFIDF_PATH)
#     print(f"✅ Saved TF-IDF vectorizer to {TFIDF_PATH}")

#     # 4️⃣ Optional semantic embeddings
#     sent_emb = compute_sentence_embeddings(texts) if USE_SENTENCE_EMB else None

#     # 5️⃣ TEXT-ONLY training
#     print("\n===== TEXT-ONLY TRAINING =====")
#     X_text_combined = combine_features(X_tfidf, np.ones(len(df)), None, sent_emb)
#     oof_text_preds, text_model = train_catboost_cv(X_text_combined, y, description="Text-only")

#     # Evaluate on original scale if price column exists
#     if "price" in df.columns:
#         smape_text = smape(df["price"].values, np.expm1(oof_text_preds))
#         print(f"\n📊 Text-only OOF SMAPE: {smape_text:.4f}%")
#     else:
#         print("⚠️ 'price' column missing — cannot compute SMAPE on original scale.")

#     joblib.dump(text_model, TEXT_MODEL_PATH)
#     print(f"✅ Saved text-only CatBoost model to {TEXT_MODEL_PATH}")

#     # 6️⃣ Multimodal training (if image embeddings exist)
#     emb = load_image_embeddings(IMAGE_EMB_PATH)
#     if emb is None:
#         print("ℹ️ Skipping multimodal training (no image embeddings).")
#         return

#     n_df, n_emb = len(df), emb.shape[0]
#     n_use = min(n_df, n_emb)
#     if n_emb != n_df:
#         print(f"⚠️ Using first {n_use} rows for multimodal training (mismatch sizes).")

#     df_sub = df.iloc[:n_use].reset_index(drop=True)
#     y_sub = df_sub["target"].values
#     X_text_sub = X_tfidf[:n_use]
#     sent_sub = sent_emb[:n_use] if sent_emb is not None else None
#     ipq = df_sub["ipq"].fillna(1).astype(float).values if "ipq" in df_sub.columns else np.ones(n_use)

#     X_multi = combine_features(X_text_sub, ipq, emb[:n_use], sent_sub)

#     print("\n===== MULTIMODAL TRAINING =====")
#     oof_multi_preds, multi_model = train_catboost_cv(X_multi, y_sub, description="Multimodal")

#     if "price" in df_sub.columns:
#         smape_multi = smape(df_sub["price"].values, np.expm1(oof_multi_preds))
#         print(f"\n📊 Multimodal OOF SMAPE: {smape_multi:.4f}%")

#     joblib.dump(multi_model, MULTI_MODEL_PATH)
#     print(f"✅ Saved multimodal CatBoost model to {MULTI_MODEL_PATH}")


# # ====================================================
# # ENTRY POINT
# # ====================================================
# if __name__ == "__main__":
#     main()

#2
# import os
# import numpy as np
# import pandas as pd
# import joblib
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.feature_extraction.text import TfidfVectorizer
# from catboost import CatBoostRegressor

# # ====================================================
# # Configuration
# # ====================================================
# DATA_PATH = "artifacts/cleaned_data.csv"
# MODEL_SAVE_PATH = "artifacts/catboost_model.pkl"
# VECTORIZER_SAVE_PATH = "artifacts/tfidf_vectorizer.pkl"
# N_SPLITS = 5
# RANDOM_STATE = 42

# # ====================================================
# # Load Data
# # ====================================================
# df = pd.read_csv(DATA_PATH)
# print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

# # Drop rows with missing essential fields
# df = df.dropna(subset=["catalog_content", "text", "target"])
# df = df.fillna("")

# # ====================================================
# # Feature Engineering
# # ====================================================
# def basic_text_features(text):
#     """Compute simple text-level numeric features."""
#     words = text.split()
#     return {
#         "word_count": len(words),
#         "char_count": len(text),
#         "avg_word_len": np.mean([len(w) for w in words]) if len(words) > 0 else 0,
#         "num_unique_words": len(set(words)),
#     }

# # Combine catalog_content and text
# df["combined_text"] = (df["catalog_content"].astype(str) + " " + df["text"].astype(str)).str.lower()

# # Add statistical text features
# text_stats = df["combined_text"].apply(basic_text_features).apply(pd.Series)
# df = pd.concat([df, text_stats], axis=1)

# # ====================================================
# # TF-IDF Features
# # ====================================================
# print("⚙️ Building TF-IDF features...")
# tfidf = TfidfVectorizer(
#     max_features=4000,
#     ngram_range=(1, 2),
#     stop_words="english",
# )

# tfidf_features = tfidf.fit_transform(df["combined_text"])
# joblib.dump(tfidf, VECTORIZER_SAVE_PATH)

# # ====================================================
# # Combine Numeric Features
# # ====================================================
# num_features = ["price", "ipq", "word_count", "char_count", "avg_word_len", "num_unique_words"]
# X_num = df[num_features].astype(float).values
# X_text = tfidf_features.toarray()

# X = np.hstack([X_text, X_num])
# y = df["target"].astype(float).values

# print(f"✅ Feature matrix: {X.shape}, Target size: {y.shape}")

# # ====================================================
# # Model Training
# # ====================================================
# cat_params = {
#     "iterations": 800,
#     "learning_rate": 0.05,
#     "depth": 8,
#     "loss_function": "MAE",
#     "random_seed": RANDOM_STATE,
#     "verbose": 100,
# }

# kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
# mae_scores, r2_scores = [], []

# fold = 1
# for train_idx, val_idx in kf.split(X):
#     print(f"\n🚀 Fold {fold}/{N_SPLITS}")
#     X_train, X_val = X[train_idx], X[val_idx]
#     y_train, y_val = y[train_idx], y[val_idx]

#     model = CatBoostRegressor(**cat_params)
#     model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

#     preds = model.predict(X_val)
#     mae = mean_absolute_error(y_val, preds)
#     r2 = r2_score(y_val, preds)

#     mae_scores.append(mae)
#     r2_scores.append(r2)

#     print(f"Fold {fold} → MAE: {mae:.4f}, R²: {r2:.4f}")
#     fold += 1

# # ====================================================
# # Final Model Training (on full data)
# # ====================================================
# print("\n✅ Training final CatBoost model on full dataset...")
# final_model = CatBoostRegressor(**cat_params)
# final_model.fit(X, y, verbose=100)

# # ====================================================
# # Save Model
# # ====================================================
# os.makedirs("artifacts", exist_ok=True)
# joblib.dump(final_model, MODEL_SAVE_PATH)

# print(f"\n🎯 Model saved to {MODEL_SAVE_PATH}")
# print(f"📊 CV MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
# print(f"📊 CV R²:  {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

#3
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from catboost import CatBoostRegressor

# ====================================================
# Configuration
# ====================================================
DATA_PATH = "artifacts/cleaned_data.csv"
IMAGE_EMB_PATH = "artifacts/image_embeddings.npy"
MODEL_SAVE_PATH = "artifacts/catboost_model.pkl"
VECTORIZER_SAVE_PATH = "artifacts/tfidf_vectorizer.pkl"
N_SPLITS = 5
RANDOM_STATE = 42

# ====================================================
# Load Data
# ====================================================
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop rows with missing essential fields
df = df.dropna(subset=["catalog_content", "text", "price"])
df = df.fillna("")

# ====================================================
# Feature Engineering
# ====================================================
def basic_text_features(text):
    words = text.split()
    return {
        "word_count": len(words),
        "char_count": len(text),
        "avg_word_len": np.mean([len(w) for w in words]) if len(words) > 0 else 0,
        "num_unique_words": len(set(words)),
    }

# Combine catalog_content and text
df["combined_text"] = (df["catalog_content"].astype(str) + " " + df["text"].astype(str)).str.lower()

# Add statistical text features
text_stats = df["combined_text"].apply(basic_text_features).apply(pd.Series)
df = pd.concat([df, text_stats], axis=1)

# ====================================================
# TF-IDF Features
# ====================================================
print("⚙️ Building TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=4000,
    ngram_range=(1, 2),
    stop_words="english",
)

tfidf_features = tfidf.fit_transform(df["combined_text"])
joblib.dump(tfidf, VECTORIZER_SAVE_PATH)
print(f"✅ TF-IDF vectorizer saved to {VECTORIZER_SAVE_PATH}")

# ====================================================
# Numeric + Text Feature Combination
# ====================================================
num_features = ["ipq", "word_count", "char_count", "avg_word_len", "num_unique_words"]
X_num = df[num_features].astype(float).values
X_text = tfidf_features.toarray()

# ====================================================
# Load Image Embeddings
# ====================================================
if os.path.exists(IMAGE_EMB_PATH):
    print("🖼️ Loading image embeddings...")
    X_img = np.load(IMAGE_EMB_PATH)
    print(f"✅ Image embeddings shape: {X_img.shape}")
    # If image count < data rows, align them
    min_len = min(len(df), len(X_img))
    X_text, X_num, X_img, y = (
        X_text[:min_len],
        X_num[:min_len],
        X_img[:min_len],
        df["price"].astype(float).values[:min_len],
    )
    # Combine all features
    X = np.hstack([X_text, X_num, X_img])
else:
    print("⚠️ No image embeddings found, training with text + numeric features only.")
    X = np.hstack([X_text, X_num])
    y = df["price"].astype(float).values

print(f"✅ Final feature matrix: {X.shape}, Target: {y.shape}")

# ====================================================
# Train/Test Split
# ====================================================
X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
    X, y, test_size=0.1, random_state=RANDOM_STATE
)
print(f"✅ Holdout set: {X_holdout.shape[0]} samples")

# ====================================================
# CatBoost Model Setup
# ====================================================
cat_params = {
    "iterations": 800,
    "learning_rate": 0.05,
    "depth": 8,
    "loss_function": "MAE",
    "random_seed": RANDOM_STATE,
    "verbose": 100,
}

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
mae_scores, r2_scores = [], []

# ====================================================
# Cross-Validation
# ====================================================
fold = 1
for train_idx, val_idx in kf.split(X_train_full):
    print(f"\n🚀 Fold {fold}/{N_SPLITS}")
    X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
    y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]

    model = CatBoostRegressor(**cat_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    mae_scores.append(mae)
    r2_scores.append(r2)

    print(f"Fold {fold} → MAE: {mae:.4f}, R²: {r2:.4f}")
    fold += 1

# ====================================================
# Final Training
# ====================================================
print("\n✅ Training final CatBoost model on full dataset...")
final_model = CatBoostRegressor(**cat_params)
final_model.fit(X, y, verbose=100)

# ====================================================
# Evaluate on Holdout
# ====================================================
print("\n🔹 Evaluating on holdout set...")
holdout_preds = final_model.predict(X_holdout)
holdout_mae = mean_absolute_error(y_holdout, holdout_preds)
holdout_r2 = r2_score(y_holdout, holdout_preds)

print(f"📊 Holdout MAE: {holdout_mae:.4f}")
print(f"📊 Holdout R²: {holdout_r2:.4f}")

# ====================================================
# Save Model
# ====================================================
os.makedirs("artifacts", exist_ok=True)
joblib.dump(final_model, MODEL_SAVE_PATH)

print(f"\n🎯 Model saved to {MODEL_SAVE_PATH}")
print(f"📊 CV MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"📊 CV R²:  {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
