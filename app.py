import streamlit as st
import os

# إعدادات الصفحة
st.set_page_config(page_title="DermAI Diagnostic Tool", page_icon="🩺")

# محاولة تحميل المكتبات الثقيلة
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import numpy as np
    from PIL import Image
    HAS_TF = True
except ImportError as e:
    HAS_TF = False
    st.error(f"Error loading dependencies: {e}")

st.title("🩺 DermAI Diagnostic Assistant")
st.markdown("---")
st.write("Upload a clinical image for preliminary analysis.")

# رفع الصورة
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and HAS_TF:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded clinical image', width=300)
    
    with st.spinner('Analyzing...'):
        try:
            # تحميل الموديل
            model = load_model('skin_disease_model.h5')
            
            # معالجة الصورة
            img = img.resize((128, 128)) 
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            
            # التوقع
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            
            st.success(f"Analysis complete. Detected Class ID: {predicted_class}")
        except Exception as e:
            st.error(f"Error during analysis: {e}")
        
    st.markdown("---")
    st.info("⚠️ **Disclaimer:** This tool is for educational purposes only. It is not a substitute for professional medical diagnosis.")

elif uploaded_file is not None and not HAS_TF:
    st.warning("System is missing required libraries to run the analysis.")
