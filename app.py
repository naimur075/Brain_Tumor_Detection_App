import streamlit as st
import cv2
import numpy as np
import h5py

class Model:
    def __init__(self):
        self.weights_input_hidden = None
        self.bias_hidden = None
        self.weights_hidden_output = None
        self.bias_output = None

    def forward(self, inputs, targets=None):
        self.hidden_layer_activation = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_activation)

        self.output_layer_activation = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = self.sigmoid(self.output_layer_activation)

        return self.output_layer_output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

def load_model_weights(model, file_path):
    with h5py.File(file_path, 'r') as file:
        # Load weights and biases
        model.weights_input_hidden = np.array(file['weights_input_hidden'])
        model.bias_hidden = np.array(file['bias_hidden'])
        model.weights_hidden_output = np.array(file['weights_hidden_output'])
        model.bias_output = np.array(file['bias_output'])

    return model  # Return the loaded model

def make_prediction_on_image(model, image_path):
    image = cv2.imdecode(np.frombuffer(image_path.read(), np.uint8), cv2.IMREAD_COLOR)
    size = (240, 240)
    resized_image = cv2.resize(image, size)
    normalized_image = (resized_image.astype(np.float32) / 255.0)
    flattened_image = normalized_image.flatten()
    data = np.expand_dims(flattened_image, axis=0)
    prediction = model.forward(data)
    return prediction

def main():
    st.title("Brain Tumor or Healthy Brain")
    st.header("Brain Tumor MRI Classifier")
    st.text("Upload a brain MRI Image for image classification as tumor or Healthy Brain")

    # Load the model
    model_file_path = r"E:\Online Class\3-2\Lab\CSE 3200\Brain-Tumor-Detection-master\models\best_model_f1_0.7578_epoch_53.h5"
    model = Model()
    model = load_model_weights(model, model_file_path)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.write("Classifying...")
        st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=True)
        label = make_prediction_on_image(model, uploaded_file)
        rounded_label = round(label[0][0], 4)
        
        if label[0] >= 0.5:  # Adjust this part according to your model's output
            st.write("Prediction Value: ", rounded_label)
            st.write("The MRI scan detects a brain tumor")
        else:
            st.write("Prediction Value: ", rounded_label)
            st.write("The MRI scan shows a healthy brain")

if __name__ == "__main__":
    main()



#c:\Users\Alvi\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages

#python -m streamlit run "E:\Online Class\3-2\Lab\CSE 3200\Brain-Tumor-Detection-master\app.py"