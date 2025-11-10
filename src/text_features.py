import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix

class TextFeatureExtractor:
    """
    Unified text feature extraction interface.
    Supports:
      1. TF-IDF Vectorization
      2. SentenceTransformer embeddings
    """

    def __init__(self, method="tfidf", max_features=10000, model_name="all-MiniLM-L6-v2"):
        """
        :param method: "tfidf" or "sentence"
        :param max_features: for TF-IDF
        :param model_name: HuggingFace model for SentenceTransformer
        """
        self.method = method
        self.max_features = max_features
        self.model_name = model_name
        self.vectorizer = None
        self.embedding_model = None

        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                stop_words="english"
            )
        elif self.method == "sentence":
            self.embedding_model = SentenceTransformer(self.model_name)
        else:
            raise ValueError("❌ Invalid method. Use 'tfidf' or 'sentence'.")

    def fit(self, texts):
        """Fit the TF-IDF model if using that method."""
        if self.method == "tfidf":
            print("🔹 Fitting TF-IDF model...")
            self.vectorizer.fit(texts)
        elif self.method == "sentence":
            print("⚙️ SentenceTransformer does not require fitting.")
        return self

    def transform(self, texts):
        """Transform text data into feature vectors."""
        print(f"🔹 Transforming text using {self.method}...")
        if self.method == "tfidf":
            X = self.vectorizer.transform(texts)
            return X
        elif self.method == "sentence":
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return csr_matrix(embeddings)

    def fit_transform(self, texts):
        """Fit and transform text data in one step."""
        if self.method == "tfidf":
            print("🔹 Building TF-IDF features...")
            return self.vectorizer.fit_transform(texts)
        elif self.method == "sentence":
            print(f"🔹 Generating embeddings with {self.model_name}...")
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return csr_matrix(embeddings)


if __name__ == "__main__":
    # Load cleaned dataset (not raw train/test)
    df = pd.read_csv("artifacts/cleaned_data.csv")
    df.columns = df.columns.str.strip()  # clean column names

    print("📄 Columns found:", df.columns.tolist())

    # Combine text fields if available
    text_columns = [c for c in ["catalog_content", "text"] if c in df.columns]
    if len(text_columns) == 0:
        raise ValueError("❌ No text columns found in cleaned_data.csv")

    df["combined_text"] = df[text_columns].astype(str).agg(" ".join, axis=1).str.lower()

    # Example: TF-IDF
    tfidf_extractor = TextFeatureExtractor(method="tfidf", max_features=10000)
    X_tfidf = tfidf_extractor.fit_transform(df["combined_text"])
    print(f"✅ TF-IDF feature matrix shape: {X_tfidf.shape}")

    # Example: Sentence embeddings (optional)
    # sentence_extractor = TextFeatureExtractor(method="sentence", model_name="all-MiniLM-L6-v2")
    # X_embed = sentence_extractor.fit_transform(df["combined_text"])
    # print(f"✅ Sentence embedding shape: {X_embed.shape}")
