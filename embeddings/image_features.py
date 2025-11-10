# """
# image_features.py
# -----------------
# Downloads product images safely and extracts embeddings using MobileNetV2 or EfficientNet.
# """

# import os
# import io
# import time
# import random
# import requests
# import numpy as np
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm

# import tensorflow as tf
# from tensorflow.keras.applications import mobilenet_v2, efficientnet
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import Model


# # ----------------------------------------------------------------
# # CONFIGURATION
# # ----------------------------------------------------------------
# IMAGE_DIR = "artifacts/images"
# EMBEDDINGS_PATH = "artifacts/image_embeddings.npy"
# MODEL_TYPE = "mobilenetv2"  # change to "efficientnet" if preferred
# IMAGE_SIZE = (224, 224)
# MAX_IMAGES = 5000  # limit to avoid rate limits in dev phase


# # ----------------------------------------------------------------
# # IMAGE DOWNLOADING
# # ----------------------------------------------------------------
# def download_image(url, retries=2, timeout=5):
#     """Download image from URL with retry and timeout handling."""
#     try:
#         headers = {"User-Agent": "Mozilla/5.0"}
#         response = requests.get(url, headers=headers, timeout=timeout)
#         response.raise_for_status()
#         img = Image.open(io.BytesIO(response.content)).convert("RGB")
#         return img
#     except Exception:
#         if retries > 0:
#             time.sleep(random.uniform(0.3, 0.8))
#             return download_image(url, retries - 1)
#         return None


# def download_images(df, url_col="product_image", limit=None):
#     """Download and save product images locally."""
#     os.makedirs(IMAGE_DIR, exist_ok=True)
#     if limit:
#         df = df.head(limit)
#     images = []
#     for i, url in tqdm(enumerate(df[url_col]), total=len(df), desc="Downloading images"):
#         img = download_image(url)
#         if img is not None:
#             img_path = os.path.join(IMAGE_DIR, f"img_{i}.jpg")
#             img.save(img_path)
#             images.append(img_path)
#     return images


# # ----------------------------------------------------------------
# # FEATURE EXTRACTION
# # ----------------------------------------------------------------
# def get_cnn_model(model_type="mobilenetv2"):
#     """Load pre-trained CNN backbone without classification head."""
#     if model_type.lower() == "mobilenetv2":
#         base_model = mobilenet_v2.MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
#     elif model_type.lower() == "efficientnet":
#         base_model = efficientnet.EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
#     else:
#         raise ValueError("Invalid model_type. Use 'mobilenetv2' or 'efficientnet'.")
#     return base_model


# def extract_image_features(image_paths, model):
#     """Generate CNN embeddings for all downloaded images."""
#     features = []
#     preprocess_input = mobilenet_v2.preprocess_input if MODEL_TYPE == "mobilenetv2" else efficientnet.preprocess_input

#     for path in tqdm(image_paths, desc="Extracting image features"):
#         try:
#             img = image.load_img(path, target_size=IMAGE_SIZE)
#             x = image.img_to_array(img)
#             x = np.expand_dims(x, axis=0)
#             x = preprocess_input(x)
#             feat = model.predict(x, verbose=0)
#             features.append(feat.squeeze())
#         except Exception as e:
#             print(f"⚠️ Error processing {path}: {e}")
#     return np.array(features)


# # ----------------------------------------------------------------
# # MAIN EXECUTION
# # ----------------------------------------------------------------
# if __name__ == "__main__":
#     train_path = "/Users/aaronrao/Desktop/screenshots/student_resource/dataset/train.csv"
#     df = pd.read_csv(train_path)

#     # Check if image URL column exists
#     url_col = None
#     for c in df.columns:
#         if "image" in c.lower():
#             url_col = c
#             break

#     if url_col is None:
#         raise ValueError("❌ No image URL column found in dataset.")

#     print(f"✅ Using image URL column: {url_col}")

#     # Step 1: Download sample images
#     image_paths = download_images(df, url_col=url_col, limit=MAX_IMAGES)

#     # Step 2: Extract embeddings
#     model = get_cnn_model(MODEL_TYPE)
#     features = extract_image_features(image_paths, model)

#     # Step 3: Save embeddings
#     os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
#     np.save(EMBEDDINGS_PATH, features)

#     print(f"✅ Saved {len(features)} image embeddings to {EMBEDDINGS_PATH}")


"""
image_features.py
-----------------
Downloads product images safely and extracts embeddings using MobileNetV2 or EfficientNet.
"""

import os
import io
import time
import random
import requests
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2, efficientnet
from tensorflow.keras.preprocessing import image


# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------
IMAGE_DIR = "artifacts/images"
EMBEDDINGS_PATH = "artifacts/image_embeddings.npy"
EMBEDDINGS_CSV_PATH = "artifacts/image_embeddings.csv"
DATA_PATH = "artifacts/cleaned_data.csv"
MODEL_TYPE = "mobilenetv2"  # or "efficientnet"
IMAGE_SIZE = (224, 224)
MAX_IMAGES = 5000  # limit for local testing


# ----------------------------------------------------------------
# IMAGE DOWNLOADING
# ----------------------------------------------------------------
def download_image(url, retries=2, timeout=5):
    """Download image from URL with retry and timeout handling."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        return img
    except Exception:
        if retries > 0:
            time.sleep(random.uniform(0.3, 0.8))
            return download_image(url, retries - 1)
        return None


def download_images(df, url_col="image_link", limit=None):
    """Download and save product images locally."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    if limit:
        df = df.head(limit)
    images = []
    for i, url in tqdm(enumerate(df[url_col]), total=len(df), desc="📥 Downloading images"):
        if not isinstance(url, str) or not url.startswith("http"):
            continue
        img = download_image(url)
        if img is not None:
            img_path = os.path.join(IMAGE_DIR, f"img_{df.loc[i, 'sample_id']}.jpg")
            img.save(img_path)
            images.append((df.loc[i, 'sample_id'], img_path))
    return images


# ----------------------------------------------------------------
# FEATURE EXTRACTION
# ----------------------------------------------------------------
def get_cnn_model(model_type="mobilenetv2"):
    """Load pre-trained CNN backbone without classification head."""
    if model_type.lower() == "mobilenetv2":
        return mobilenet_v2.MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    elif model_type.lower() == "efficientnet":
        return efficientnet.EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
    else:
        raise ValueError("Invalid model_type. Use 'mobilenetv2' or 'efficientnet'.")


def extract_image_features(image_pairs, model):
    """Generate CNN embeddings for all downloaded images."""
    features, ids = [], []
    preprocess_input = mobilenet_v2.preprocess_input if MODEL_TYPE == "mobilenetv2" else efficientnet.preprocess_input

    for sample_id, path in tqdm(image_pairs, desc="🔍 Extracting image features"):
        try:
            img = image.load_img(path, target_size=IMAGE_SIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feat = model.predict(x, verbose=0)
            features.append(feat.squeeze())
            ids.append(sample_id)
        except Exception as e:
            print(f"⚠️ Error processing {path}: {e}")
    return np.array(ids), np.array(features)


# ----------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    print(f"📘 Loaded cleaned data: {df.shape}")

    # Detect image URL column
    url_col = None
    for c in df.columns:
        if "image" in c.lower():
            url_col = c
            break
    if url_col is None:
        raise ValueError("❌ No image URL column found in dataset.")
    print(f"✅ Using image URL column: {url_col}")

    # Step 1: Download sample images
    image_pairs = download_images(df, url_col=url_col, limit=MAX_IMAGES)

    # Step 2: Extract embeddings
    model = get_cnn_model(MODEL_TYPE)
    ids, features = extract_image_features(image_pairs, model)

    # Step 3: Save embeddings
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    np.save(EMBEDDINGS_PATH, features)

    df_emb = pd.DataFrame(features)
    df_emb.insert(0, "sample_id", ids)
    df_emb.to_csv(EMBEDDINGS_CSV_PATH, index=False)

    print(f"✅ Saved {len(features)} embeddings to:")
    print(f"   → {EMBEDDINGS_PATH}")
    print(f"   → {EMBEDDINGS_CSV_PATH}")
