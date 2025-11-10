import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import nltk

# Download basic NLTK resources (if not already)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """
    Basic text cleaning function:
    - Lowercase
    - Remove special characters and extra spaces
    - Remove stopwords
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)


def extract_ipq(text: str) -> int:
    """
    Extract Item Pack Quantity (IPQ) from text if present.
    Example: 'Pack of 3', '3 pcs', '2 units' → returns 3, 3, 2
    """
    if not isinstance(text, str):
        return 1
    match = re.search(r'(\d+)\s*(pack|pcs?|pieces?|units?)', text.lower())
    return int(match.group(1)) if match else 1


def load_and_preprocess(train_path: str, test_path: str):
    """
    Load train and test datasets, clean text, extract IPQ feature.
    Returns processed DataFrames.
    """
    print("🔹 Loading datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    print("🔹 Cleaning catalog content and extracting IPQ...")
    for df in [train_df, test_df]:
        df['clean_text'] = df['catalog_content'].apply(clean_text)
        df['ipq'] = df['catalog_content'].apply(extract_ipq)

    # Optional: log-transform price to stabilize scale
    if 'price' in train_df.columns:
        train_df['log_price'] = train_df['price'].apply(lambda x: np.log1p(x))

    return train_df, test_df


def split_data(train_df, test_size=0.1, random_state=42):
    """
    Split training data for local validation.
    """
    print("🔹 Splitting train and validation data...")
    train, val = train_test_split(train_df, test_size=test_size, random_state=random_state)
    return train, val


if __name__ == "__main__":
    train_path = "/Users/aaronrao/Desktop/screenshots/student_resource/dataset/train.csv"
    test_path = "/Users/aaronrao/Desktop/screenshots/student_resource/dataset/test.csv"

    train_df, test_df = load_and_preprocess(train_path, test_path)
    train, val = split_data(train_df)

    print("✅ Data preprocessing completed!")
    print(train.head())

    # Rename columns to what model_training.py expects
    train_df = train_df.rename(columns={
        "clean_text": "text",
        "log_price": "target"
    })

    # Save preprocessed data
    os.makedirs("artifacts", exist_ok=True)
    train_df.to_csv("artifacts/cleaned_data.csv", index=False)
    print("✅ Saved preprocessed data to artifacts/cleaned_data.csv")
