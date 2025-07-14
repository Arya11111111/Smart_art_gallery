  Smart Image Insight App

An AI-powered web application that analyzes uploaded paintings or images to:
Predict their category using deep learning  logistic regression.
Describe the image in artistic detail via multimodal LLM (LLaVA).
Recommend visually similar images using FAISS.

---

  Features
ResNet50 for deep feature extraction
Logistic Regression for classification
FAISS for similar image retrieval
LangChain  Ollama (LLaVA) for descriptive text generation
Streamlit frontend with FastAPI backend

---

  Technologies
Python, TensorFlow, Streamlit, FastAPI
FAISS, scikit-learn, joblib
LangChain, Ollama (LLaVA)
PIL, NumPy, Uvicorn

---

  Project Structure

```
 project/
 streamlit_app.py                Frontend
 server.py                       FastAPI backend
 train_model.py                  Model training  FAISS index
 test_model.py                   Manual test script
 assets/                         App banners  UI images
 images/images/                  Dataset folder (images organized by category)
 faiss_index.bin                 Saved FAISS index
 image_classifier.pkl            Trained classifier
 image_paths.pkl                 Saved image paths
 requirements.txt
 README.md
```

---

  Setup Instructions

 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Also ensure Ollama(https://ollama.com) is installed and running with the `llava` model.

```bash
ollama run llava
```

---

 2. Train the Model

```bash
python train_model.py
```
Extracts ResNet50 features
Trains logistic regression classifier
Builds FAISS index

---

 3. Start Backend Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

---

 4. Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

Upload an image  click  Generate Insights  receive predictions, description, and similar artworks.

---

  Notes
Folder names are used as class labels.
Descriptions are AI-generated using LangChain and the LLaVA model.
Ollama must be running in the background for image description.

---

  License

This project is for educational and demonstration purposes only.
