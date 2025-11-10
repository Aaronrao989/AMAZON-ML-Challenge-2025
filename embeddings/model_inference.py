import os
import joblib
import numpy as np
import pandas as pd
from catboost import Pool

# Define base directory and paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACT_DIR, "catboost_model.pkl")
VECTORIZER_PATH = os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl")
IMAGE_EMB_PATH = os.path.join(ARTIFACT_DIR, "image_embeddings.csv")

def load_artifacts():
    """Load model and required artifacts with error handling"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vectorizer not found at {VECTORIZER_PATH}")
    
    print("🔍 Loading model and TF-IDF vectorizer...")
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
    
    image_emb = None
    if os.path.exists(IMAGE_EMB_PATH):
        print("📸 Loading image embeddings...")
        image_emb = pd.read_csv(IMAGE_EMB_PATH)
    else:
        print("⚠️ No image embeddings found, will proceed without image features")
    
    return model, tfidf, image_emb

def basic_text_features(text):
    words = text.split()
    return {
        "word_count": len(words),
        "char_count": len(text),
        "avg_word_len": np.mean([len(w) for w in words]) if len(words) > 0 else 0,
        "num_unique_words": len(set(words)),
    }

def preprocess_input(df, tfidf, image_emb=None):
    """Preprocess input data with consistent feature structure"""
    try:
        df = df.copy()
        
        # 1. Text Features - ensure exact TF-IDF vocabulary
        df = df.fillna("")
        text_col = df["text"] if "text" in df.columns else pd.Series([""] * len(df))
        catalog_col = df["catalog_content"] if "catalog_content" in df.columns else pd.Series([""] * len(df))
        df["combined_text"] = (catalog_col.astype(str) + " " + text_col.astype(str)).str.lower()

        # Get TF-IDF features with exact vocabulary
        X_text = tfidf.transform(df["combined_text"]).toarray()
        print(f"TF-IDF features shape: {X_text.shape}")

        # 2. Numeric Features
        num_features = ["ipq", "word_count", "char_count", "avg_word_len", "num_unique_words"]
        text_stats = df["combined_text"].apply(basic_text_features).apply(pd.Series)
        for col in num_features:
            if col not in df.columns:
                df[col] = text_stats[col] if col in text_stats else 0
        X_num = df[num_features].astype(float).values
        print(f"Numeric features shape: {X_num.shape}")

        # 3. Image Features
        if image_emb is not None:
            print("Adding image embeddings...")
            # Get numeric columns excluding sample_id
            emb_cols = [col for col in image_emb.columns if col != 'sample_id']
            
            # Merge embeddings
            df = df.merge(image_emb, on="sample_id", how="left")
            df[emb_cols] = df[emb_cols].fillna(0)
            X_img = df[emb_cols].values
            print(f"Image features shape: {X_img.shape}")
            
            X = np.hstack([X_text, X_num, X_img])
        else:
            X = np.hstack([X_text, X_num])

        print(f"\nFeature counts:")
        print(f"- TF-IDF features: {X_text.shape[1]}")
        print(f"- Numeric features: {len(num_features)}")
        print(f"- Image features: {len(emb_cols) if image_emb is not None else 0}")
        print(f"Total features: {X.shape[1]}")
        
        return X

    except Exception as e:
        print(f"❌ Preprocessing error: {str(e)}")
        raise

def predict_from_csv(input_path, output_path=None):
    """Generate predictions with feature validation"""
    try:
        # Load model and artifacts
        model, tfidf, image_emb = load_artifacts()
        
        print(f"📂 Reading input data from {input_path}")
        df = pd.read_csv(input_path)
        
        print("⚡ Preprocessing input data...")
        X = preprocess_input(df, tfidf, image_emb)
        
        # Validate feature dimensions
        validate_features(model, X)
        
        print("🔮 Generating predictions...")
        pool = Pool(X)
        preds = model.predict(pool)
        
        # Create and save predictions
        preds_df = pd.DataFrame({
            "sample_id": df["sample_id"],
            "predicted_target": preds
        })
        
        if output_path is None:
            output_path = os.path.join(ARTIFACT_DIR, "predictions.csv")
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        preds_df.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to {output_path}")
        
        return preds_df

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

def validate_features(model, X):
    """Debug feature dimensions"""
    try:
        # Get feature count for CatBoost model
        if hasattr(model, 'get_feature_count'):
            model_features = model.get_feature_count()
        else:
            # Fallback to checking model parameters
            model_features = model.tree_count_ * model.get_params().get('depth', 6)
            
        input_features = X.shape[1]
        
        print("\n🔍 Feature validation:")
        print(f"- Model expects: {model_features} features")
        print(f"- Input has: {input_features} features")
        
        if model_features != input_features:
            raise ValueError(
                f"\nFeature count mismatch!"
                f"\n- Model expects: {model_features} features"
                f"\n- Input has: {input_features} features"
                f"\n\nPlease check:"
                f"\n1. TF-IDF vocabulary matches training"
                f"\n2. All numeric features are present"
                f"\n3. Feature ordering is consistent"
            )
    except Exception as e:
        print(f"⚠️ Warning: Feature validation skipped - {str(e)}")
        # Continue without validation
        pass

if __name__ == "__main__":
    try:
        input_path = os.path.join(ARTIFACT_DIR, "cleaned_data.csv")
        preds_df = predict_from_csv(input_path)
        print("\n📊 Preview of predictions:")
        print(preds_df.head())
    except Exception as e:
        print(f"❌ Script failed: {str(e)}")
