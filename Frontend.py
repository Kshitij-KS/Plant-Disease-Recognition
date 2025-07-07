import streamlit as st
import tensorflow as tf
import keras
import numpy as np
import cv2
import tempfile

@st.cache_resource
def load_model():
    return keras.models.load_model(r"plantdiseases_model.keras")

def model_predict(image_path, model):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image at path: {image_path}")
    H, W, C = 224, 224, 3
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = img.reshape(1, H, W, C)
    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

if 'page' not in st.session_state:
    st.session_state.page = 'HOME'

def switch_page():
    if st.session_state.page == 'HOME':
        st.session_state.page = 'DISEASE RECOGNITION'
    else:
        st.session_state.page = 'HOME'

model = load_model()

if st.session_state.page == 'HOME':
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)
    st.image("Plant-Disease-Title.jpg", caption="Plant Disease Detection", use_container_width=True)
    if st.button('Go to Disease Recognition'):
        switch_page()

elif st.session_state.page == 'DISEASE RECOGNITION':
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    if st.button('Back to Home'):
        switch_page()
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        # Save to a temporary file to avoid overwriting
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(test_image.getbuffer())
            tmp_path = tmp_file.name
        st.image(test_image, use_container_width=True)
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            result_index = model_predict(tmp_path, model)
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            st.success(f"Model is Predicting it's a {class_name[result_index]}")
