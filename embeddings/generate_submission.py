import os
import pandas as pd
import numpy as np
import joblib
from catboost import Pool

# ====================================================
# Configuration
# ====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts/catboost_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "artifacts/tfidf_vectorizer.pkl")
TEST_DATA_PATH = os.path.join(BASE_DIR, "dataset/test.csv")
SUBMISSION_PATH = os.path.join(BASE_DIR, "artifacts/submission.csv")
IMAGE_EMB_PATH = os.path.join(BASE_DIR, "artifacts/image_embeddings.csv")

NUMERIC_FEATURES = ["ipq", "word_count", "char_count", "avg_word_len", "num_unique_words"]

# ====================================================
# Basic text features
# ====================================================
def basic_text_features(text):
    words = text.split()
    return {
        "word_count": len(words),
        "char_count": len(text),
        "avg_word_len": np.mean([len(w) for w in words]) if len(words) > 0 else 0,
        "num_unique_words": len(set(words)),
    }

# ====================================================
# Preprocess test data
# ====================================================
def preprocess_input(df, tfidf, image_emb=None):
    """Preprocess input data with consistent feature structure"""
    try:
        df = df.copy()
        
        # 1. Text Features - TF-IDF
        text_col = df["text"] if "text" in df.columns else pd.Series([""] * len(df))
        catalog_col = df["catalog_content"] if "catalog_content" in df.columns else pd.Series([""] * len(df))
        df["combined_text"] = (catalog_col.astype(str) + " " + text_col.astype(str)).str.lower()
        
        X_text = tfidf.transform(df["combined_text"]).toarray()
        print(f"TF-IDF features shape: {X_text.shape}")

        # 2. Numeric Features
        text_stats = df["combined_text"].apply(basic_text_features).apply(pd.Series)
        for col in NUMERIC_FEATURES:
            if col not in df.columns:
                df[col] = text_stats[col] if col in text_stats else 0
        X_num = df[NUMERIC_FEATURES].astype(float).values
        print(f"Numeric features shape: {X_num.shape}")

        # 3. Image Features
        if image_emb is not None:
            print("Adding image embeddings...")
            emb_cols = [col for col in image_emb.columns if col != 'sample_id']
            df = df.merge(image_emb, on="sample_id", how="left")
            df[emb_cols] = df[emb_cols].fillna(0)
            X_img = df[emb_cols].values
            print(f"Image features shape: {X_img.shape}")
            X = np.hstack([X_text, X_num, X_img])
        else:
            X = np.hstack([X_text, X_num])

        print(f"\nFeature counts:")
        print(f"- TF-IDF features: {X_text.shape[1]}")
        print(f"- Numeric features: {len(NUMERIC_FEATURES)}")
        print(f"- Image features: {len(emb_cols) if image_emb is not None else 0}")
        print(f"Total features: {X.shape[1]}")
        
        return X

    except Exception as e:
        print(f"❌ Preprocessing error: {str(e)}")
        raise

# ====================================================
# Generate submission
# ====================================================
def generate_submission():
    print("📂 Reading test data from", TEST_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)

    print("🔍 Loading trained model and TF-IDF vectorizer...")
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
    
    # Load image embeddings if available
    image_emb = None
    if os.path.exists(IMAGE_EMB_PATH):
        print("📸 Loading image embeddings...")
        image_emb = pd.read_csv(IMAGE_EMB_PATH)

    print("⚡ Preprocessing test data...")
    X_test = preprocess_input(df_test, tfidf, image_emb)

    print("📝 Making predictions...")
    # Convert to Pool for CatBoost
    test_pool = Pool(X_test)
    predicted_price = model.predict(test_pool)

    submission = pd.DataFrame({
        "sample_id": df_test["sample_id"],
        "price": predicted_price
    })

    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"✅ Submission saved to {SUBMISSION_PATH}")
    print("\nPreview of predictions:")
    print(submission.head())

# ====================================================
# Run
# ====================================================
if __name__ == "__main__":
    generate_submission()
