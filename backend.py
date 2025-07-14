from fastapi import FastAPI, File, UploadFile
import numpy as np
import faiss
import joblib
import uvicorn
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from langchain_community.llms import Ollama

app = FastAPI()

# 🔹 Load FAISS Index and Image Paths
index = faiss.read_index("faiss_index.bin")
img_paths = joblib.load("image_paths.pkl")

# 🔹 Load ResNet50 model for feature extraction
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# 🔹 Feature Extraction Function
def extract_features(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = resnet_model.predict(img_array)
    return features.flatten()

# 🔹 Find Similar Images
def find_similar_images(image_features, k=3):
    image_features = np.array([image_features]).astype('float32')
    _, indices = index.search(image_features, k)
    return [img_paths[i] for i in indices[0]]

# 🔹 Describe Image with Ollama (LLaVA)
def describe_image(img):
    llm = Ollama(model="llava")
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    prompt = """
    Analyze and describe this painting with artistic depth:
    - Identify the artistic style (e.g., Impressionism, Realism, Abstract, Baroque)
    - Describe the subject, background, and artistic elements in detail
    - Explain the dominant color palette and mood
    - Provide a possible interpretation and historical significance
    - Mention if it resembles any famous artist’s work
    """

    response = llm.invoke(prompt, images=[img_b64])
    return response

# 🔹 FastAPI Endpoint
@app.post("/process/")
async def process_image(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read())).convert("RGB")

    # Extract image features
    features = extract_features(img)

    # 🔁 Predict label from folder name of most similar image
    most_similar = find_similar_images(features, k=1)[0]
    prediction = most_similar.split("\\")[-2]  # Folder name is the label

    # Describe the image
    description = describe_image(img)

    # Get top 3 similar images
    similar_images = find_similar_images(features)

    return {
        "prediction": prediction,
        "description": description,
        "similar_images": similar_images
    }

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
