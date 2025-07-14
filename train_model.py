import os
import faiss
import numpy as np
import joblib
import pickle
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.linear_model import LogisticRegression

# ✅ Check if models already exist before retraining
if os.path.exists("faiss_index.bin") and os.path.exists("image_classifier.pkl") and os.path.exists("image_paths.pkl"):
    print("✅ FAISS index and model already exist. Loading instead of retraining...")

    # Load FAISS index
    index = faiss.read_index("faiss_index.bin")

    # Load trained model
    model = joblib.load("image_classifier.pkl")

    # Load image paths
    img_paths = joblib.load("image_paths.pkl")

    print("✅ FAISS index and model loaded successfully!")
    exit()  # 🚀 Skip retraining

# Load pre-trained ResNet50 model for feature extraction
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Path to dataset
DATASET_PATH = "images/images"

# Function to extract deep features
def extract_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = resnet_model.predict(img_array)
    return features.flatten()

# Load dataset and extract features
X, y, img_paths = [], [], []
categories = os.listdir(DATASET_PATH)

for category in categories:
    category_path = os.path.join(DATASET_PATH, category)
    if os.path.isdir(category_path):
        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)
            X.append(extract_features(img_path))
            y.append(category)
            img_paths.append(img_path)

X = np.array(X)
y = np.array(y)

# Train a Logistic Regression model for classification
model = LogisticRegression(max_iter=1000)
model.fit(X, y)


# Save classification model
joblib.dump(model, "image_classifier.pkl")

# **FAISS Indexing**
dimension = X.shape[1]  # Feature vector size
index = faiss.IndexFlatL2(dimension)  # L2 distance index
index.add(X)  # Add features to FAISS index

# Save FAISS index & image paths
faiss.write_index(index, "faiss_index.bin")
joblib.dump(img_paths, "image_paths.pkl")

print("✅ Model training complete! FAISS index and image classifier saved.")

# **Test FAISS Loading to Confirm**
index = faiss.read_index("faiss_index.bin")  # ✅ Correct way to load FAISS
model = joblib.load("image_classifier.pkl")  # ✅ Correct way to load model
img_paths = joblib.load("image_paths.pkl")  # ✅ Load image paths

print("✅ FAISS index and model loaded successfully!")
