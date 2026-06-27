import streamlit as st
import numpy as np
from PIL import Image
import sys
import subprocess

# دالة لتثبيت المكتبة إذا لم تكن موجودة
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# إعدادات الصفحة
st.set_page_config(page_title="DermAI Diagnostic Tool", page_icon="🩺")

# محاولة تحميل Tensorflow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
except ImportError:
    st.warning("جاري تجهيز بيئة العمل، يرجى الانتظار دقيقة...")
    install("tensorflow")
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image

# دالة لتحميل الموديل بطريقة ذكية (Caching)
@st.cache_resource
def get_model():
    # تأكدي من اسم الملف بالظبط في GitHub
    return load_model('skin_disease_model(1).h5')

st.title("🩺 DermAI Diagnostic Assistant")
st.markdown("---")
st.write("Upload a clinical image for preliminary analysis.")

# رفع الصورة
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded clinical image', width=300)
    
    with st.spinner('Analyzing...'):
        try:
            # استدعاء الموديل
            model = get_model()
            
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
