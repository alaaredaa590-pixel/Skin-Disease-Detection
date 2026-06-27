import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(page_title="DermAI Diagnostic Tool", page_icon="🩺")

try:
    model = load_model('skin_disease_model.h5')
except:
    st.error("Error: Model file not found.")

st.title(" DermAI Diagnostic Assistant")
st.markdown("---")
st.write("Upload a clinical image for preliminary analysis.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded clinical image', width=300)
    
    with st.spinner('Analyzing image...'):
      
        img = img.resize((128, 128)) 
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        
        st.success(f"Analysis complete. Detected Class ID: {predicted_class}")
        
    st.markdown("---")
    st.info(" **Disclaimer:** This tool is for educational purposes and provides preliminary data only. It is not a substitute for professional medical diagnosis. Please consult a dermatologist.")
