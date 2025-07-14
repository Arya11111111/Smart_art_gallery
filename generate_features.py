import os
import joblib
import numpy as np
import faiss
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import unicodedata

# ✅ Load ResNet50 model
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# ✅ Corrected File Path (use raw string)
image_folder = r"C:\Users\Arya Patil\OneDrive\Desktop\smart_image_app\images\images"

# ✅ Check if folder exists
if not os.path.exists(image_folder):
    print(f"❌ Folder not found: {image_folder}")
    exit()

# ✅ Function to normalize Unicode paths (Fix special character issues)
def normalize_unicode_path(path):
    return unicodedata.normalize("NFC", path)

# ✅ List all images in the folder
image_paths = []
for root, _, files in os.walk(image_folder):
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            image_paths.append(normalize_unicode_path(os.path.join(root, file)))

if not image_paths:
    print("⚠️ No images found! Check if the folder contains valid images.")
    exit()
else:
    print(f"✅ Found {len(image_paths)} images.")

# ✅ Function to extract features
def extract_features(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = resnet_model.predict(img_array)
        return features.flatten()
    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None

# ✅ Extract features for all images
image_features = [extract_features(img_path) for img_path in image_paths]
image_features = [feat for feat in image_features if feat is not None]  # Remove failed ones

# ✅ Convert to NumPy array
if image_features:
    image_features = np.array(image_features).astype('float32')
    joblib.dump(image_features, "image_features.pkl")
    joblib.dump(image_paths, "image_paths.pkl")

    # ✅ Only create FAISS index if features exist
    if image_features.shape[0] > 0:
        index = faiss.IndexFlatL2(image_features.shape[1])
        index.add(image_features)
        faiss.write_index(index, "faiss_index.bin")

        print("✅ Image features and FAISS index saved successfully!")
    else:
        print("⚠️ No valid features extracted! Skipping FAISS index creation.")
else:
    print("⚠️ No valid features extracted! Check the image files.")
