import joblib
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load trained model
model = joblib.load("image_classifier.pkl")

# Load pre-trained ResNet50 model for feature extraction
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Function to extract deep features from an image
def extract_features(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Resize image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = resnet_model.predict(img_array)
    return features.flatten()

# Test on a new image
test_image_path = "test.jpg"  # Replace this with an actual image path
features = extract_features(test_image_path)

# Predict category
prediction = model.predict([features])
print(f"🎨 Predicted Category: {prediction[0]}")
