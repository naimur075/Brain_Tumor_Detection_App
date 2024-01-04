import streamlit as st
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import keras

def teachable_machine_classification(img, weights_file):
    
    model = keras.models.load_model(weights_file)

    
    data = np.ndarray(shape=(1, 240, 240, 3), dtype=np.float32)
    image = img
  
    size = (240, 240)
    image = image.resize(size, Image.ANTIALIAS)

  
    image_array = np.asarray(image)
  
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

  
    data[0] = normalized_image_array

    prediction = model.predict(data)
    return np.argmax(prediction)

st.title("Brain Tumor or Healthy Brain")
st.header("Brain Tumor MRI Classifier")
st.text("Upload a brain MRI Image for image classification as tumor or Healthy Brain")
     
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    
    st.write("")
    label = teachable_machine_classification(image, 'best_model_f1_0.7660_epoch_99.h5')
    if label == 0:
       st.write("The MRI scan detects a brain tumor")
    else:
       st.write("The MRI scan shows an healthy brain")
   
        
        
