import streamlit as st
import requests
from PIL import Image



# 🔹 Apply background image with gradient overlay
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://i.pinimg.com/736x/72/44/0f/72440f6b77a824e1b429a62d5c53622d.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .gradient-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(to bottom, rgba(0, 0, 0, 0.7), rgba(255, 255, 255, 0.1));
        z-index: 1;
    }
    .info-box {
        background: #5C4033;  /* Rich brown for better contrast */
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        margin-bottom: 15px;
        color: #FFFFFF;  /* White text for better readability */
        font-size: 16px;
        font-weight: bold;
    }
    .center-title {
        text-align: center;
        color: #FF5733;
        font-weight: bold;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.5);
    }
    </style>
    <div class="gradient-overlay"></div>
    """,
    unsafe_allow_html=True
)

# 🎨 Custom title banner
title_image = Image.open("assets/smart_image_title.png")
st.image(title_image, use_column_width=True)

# 📸 Upload an image
uploaded_image = st.file_uploader("📸 Upload an image", type=["jpg", "png", "jpeg"])

# FastAPI backend URL (Update this as needed)
API_URL = "http://localhost:8000/process/"

if uploaded_image:
    st.image(uploaded_image, caption="📷 Uploaded Image", use_column_width=True)

    if st.button("🔍 Generate Insights"):
        with st.spinner("🔄 Processing..."):
            try:
                # Convert uploaded image to bytes
                img_bytes = uploaded_image.getvalue()

                # Send image to FastAPI backend
                response = requests.post(
                    API_URL,
                    files={"file": ("image.jpg", img_bytes, uploaded_image.type)}
                )

                # Process Response
                if response.status_code == 200:
                    result = response.json()
                    
                    # 🎯 Extract Data
                    prediction = result.get("prediction", "Unknown Category")
                    description = result.get("description", "No detailed description available.")
                    similar_images = result.get("similar_images", [])

                    # 🎨 **Display Results**
                    st.markdown(f"""
                    <div class='info-box'>
                        <h3>🎨 Predicted Category:</h3>
                        <p>{prediction}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class='info-box'>
                        <h3>📝 AI-Generated Description:</h3>
                        <p>{description}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # 🔍 **Display Similar Images**
                    if similar_images:
                        st.markdown("<h3 class='center-title'>🔍 Similar Paintings:</h3>", unsafe_allow_html=True)
                        cols = st.columns(min(len(similar_images), 4))  # Show up to 4 images per row
                        for i, img_url in enumerate(similar_images):
                            with cols[i % len(cols)]:  # Distribute images across columns
                                st.image(img_url, caption=f"Similar Image {i+1}", use_column_width=True)
                    else:
                        st.info("No similar images found.")

                else:
                    st.error("🚨 Error processing image. Please try again.")

            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the backend. Make sure the FastAPI server is running.")
