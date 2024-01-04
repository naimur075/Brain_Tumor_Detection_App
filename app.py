import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Function to preprocess and classify the image
def teachable_machine_classification(img, model):
    size = (240, 240)
    image = img.resize(size, Image.ANTIALIAS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data = np.ndarray(shape=(1, 240, 240, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    return np.argmax(prediction)

def main():
    st.title("Brain Tumor or Healthy Brain")
    st.header("Brain Tumor MRI Classifier")
    st.text("Upload a brain MRI Image for image classification as tumor or Healthy Brain")

    # Load the model
    model_file_path = 'best_model_f1_0.7660_epoch_99.h5'
    model = load_model(model_file_path)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")

        # Classify the image
        label = teachable_machine_classification(image, model)

        # Display the result
        if label == 0:
            st.write("The MRI scan detects a brain tumor")
        else:
            st.write("The MRI scan shows a healthy brain")

if __name__ == "__main__":
    main()
