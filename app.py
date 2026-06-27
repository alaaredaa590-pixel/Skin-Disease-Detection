import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 1. إعدادات الصفحة
st.set_page_config(page_title="DermAI Diagnostic Tool", page_icon="🩺")

# 2. تحميل الموديل بطريقة الـ Caching لزيادة السرعة
@st.cache_resource
def get_model():
    # تأكدي أن اسم الملف في GitHub يطابق هذا الاسم تماماً
    return load_model('skin_disease_model(1).h5')

# محاولة تحميل الموديل مع معالجة الأخطاء
try:
    model = get_model()
except Exception as e:
    st.error(f"خطأ في تحميل الموديل: {e}")
    st.stop()

# 3. واجهة المستخدم
st.title("🩺 DermAI Diagnostic Assistant")
st.markdown("---")
st.write("قومي برفع صورة لحالة جلدية لتحليلها.")

# 4. رفع الصورة
uploaded_file = st.file_uploader("اختاري صورة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة المرفوعة
    img = Image.open(uploaded_file)
    st.image(img, caption='الصورة المرفوعة', width=300)
    
    with st.spinner('جاري التحليل...'):
        try:
            # 5. معالجة الصورة لتناسب الموديل
            img = img.resize((128, 128)) 
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0 # تطبيع البيانات (Normalization)
            
            # 6. التوقع
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            
            # عرض النتيجة
            st.success(f"تم التحليل بنجاح. التصنيف هو: {predicted_class}")
        except Exception as e:
            st.error(f"حدث خطأ أثناء التحليل: {e}")
        
    st.markdown("---")
    st.info("⚠️ **تنبيه:** هذه الأداة للأغراض التعليمية فقط، وليست بديلاً عن التشخيص الطبي المتخصص.")
