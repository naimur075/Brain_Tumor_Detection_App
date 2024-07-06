import streamlit as st
import cv2
import numpy as np
import time
import os
import shutil

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))
        self.loss_function = None

    def sigmoid(self, x):
        clipped_x = np.clip(x, -500, 500)  
        return 1 / (1 + np.exp(-clipped_x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.hidden_layer_activation = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_activation)

        self.output_layer_activation = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = self.sigmoid(self.output_layer_activation)

        return self.output_layer_output

    def backward(self, inputs, targets, learning_rate):
        output_error = targets - self.output_layer_output
        output_delta = output_error * self.sigmoid_derivative(self.output_layer_output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate 

    def compile(self, optimizer, loss_function, metrics=None):
        # Store optimizer and loss function
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.metrics = metrics

    def save_model(self, folder_path, model_count):
        architecture = {
            'input_size': self.weights_input_hidden.shape[0],
            'hidden_size': self.weights_input_hidden.shape[1],
            'output_size': self.weights_hidden_output.shape[1]
        }

        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            file_path = os.path.join(folder_path, f"trained_model_{model_count}.npz")  
            np.savez(file_path, architecture=architecture, 
                     weights_input_hidden=self.weights_input_hidden,
                     weights_hidden_output=self.weights_hidden_output)

            print(f"Model saved at: {file_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

def train_neural_network(X_train, y_train, num_epochs, learning_rate, new_train):
    folder_path = "Real Time Model"

    if new_train:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        model_count = 1
    else:
        model_count = len([f for f in os.listdir(folder_path) if f.startswith("trained_model_")]) + 1

    input_size = X_train.shape[1]
    hidden_size = 128
    output_size = 1

    model = NeuralNetwork(input_size, hidden_size, output_size)

    optimizer = "adam"
    loss_function = "binary_crossentropy"
    metrics = ["accuracy"]
    model.compile(optimizer, loss_function, metrics)

    start_time = time.time()

    for epoch in range(num_epochs):
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
        train_predictions = model.forward(X_train)
        loss = np.mean(np.square(y_train - train_predictions))
        model.backward(X_train, y_train, learning_rate)

        if epoch % 10 == 0:
            st.write(f"Epoch {epoch}/{num_epochs} - Loss: {loss:.4f}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write(f"Training completed in {elapsed_time} seconds.")

    model.save_model(folder_path, model_count)

    return model

def load_latest_model(folder_path):
    model_files = [f for f in os.listdir(folder_path) if f.startswith("trained_model_")]
    if not model_files:
        st.error("No trained models found.")
        return None

    latest_model_file = max(model_files)
    file_path = os.path.join(folder_path, latest_model_file)

    print(f"Loading latest model from: {file_path}")

    try:
        with np.load(file_path, allow_pickle=True) as f:
            architecture = f['architecture'].item()
            weights_input_hidden = f['weights_input_hidden']
            weights_hidden_output = f['weights_hidden_output']

        input_size = architecture['input_size']
        hidden_size = architecture['hidden_size']
        output_size = architecture['output_size']

        model = NeuralNetwork(input_size, hidden_size, output_size)

        model.weights_input_hidden = weights_input_hidden
        model.weights_hidden_output = weights_hidden_output

        return model, latest_model_file
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def make_prediction_on_image(model, uploaded_file):
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    size = (240, 240)
    resized_image = cv2.resize(image, size)
    normalized_image = (resized_image.astype(np.float32) / 255.0)
    flattened_image = normalized_image.flatten()
    
    if len(flattened_image) < 57600:
        flattened_image = np.pad(flattened_image, (0, 57600 - len(flattened_image)), mode='constant')
    elif len(flattened_image) > 57600:
        flattened_image = flattened_image[:57600]
    
    data = flattened_image.reshape(1, -1) 
    prediction = model.forward(data)
    return prediction

def main():
    st.title("Brain Tumor or Healthy Brain")
    st.header("Brain Tumor MRI Classifier")
    
    option = st.radio(
        'Select an option:',
        ('Train', 'Test')
    )

    if option == 'Train':
        num_types = st.selectbox("How many types?", [2, 1])
        
        st.text("Upload brain MRI images for training")
        uploaded_files = st.file_uploader("Choose brain MRI images...", accept_multiple_files=True)

        labels = []
        if uploaded_files is not None and len(uploaded_files) > 0:
            for uploaded_file in uploaded_files:
                st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=True)
                label = st.selectbox(f"Label for {uploaded_file.name}", options=[0, 1], key=uploaded_file.name)
                labels.append((uploaded_file, label))
            
            if len(set(label for _, label in labels)) != num_types:
                st.warning(f"Please ensure the number of unique labels matches the selected types ({num_types}).")
            else:
                train_option = st.radio("Select training type:", ('New Train', 'Existing Train'))
                start_training = st.button("Start Training")
                if start_training:
                    X_train = []
                    y_train = []

                    for uploaded_file, label in labels:
                        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                        resized_image = cv2.resize(image, (240, 240))
                        normalized_image = resized_image.astype(np.float32) / 255.0
                        flattened_image = normalized_image.flatten()
                        X_train.append(flattened_image)
                        y_train.append(label)

                    X_train = np.array(X_train)
                    y_train = np.array(y_train).reshape(-1, 1)

                    input_size = X_train.shape[1]
                    num_epochs = 100
                    learning_rate = 0.001

                    new_train = train_option == 'New Train'
                    model = train_neural_network(X_train, y_train, num_epochs, learning_rate, new_train)
                    st.write("Training completed!")

    if option == 'Test':
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            model, latest_model_file = load_latest_model("Real Time Model")
            if model is not None and latest_model_file is not None:
                st.write(f"Loading latest trained model: {latest_model_file}...")
                st.write("\nClassifying...")
                st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=True)

                label = make_prediction_on_image(model, uploaded_file)
                rounded_label = round(label[0][0], 4)

                if label[0] > 0.5:  
                    st.write("Prediction Value: ", rounded_label)
                    st.write("The MRI scan detects a brain tumor")
                else:
                    st.write("Prediction Value: ", rounded_label)
                    st.write("The MRI scan shows a healthy brain")

if __name__ == "__main__":
    main()


# New Train, Existing Train Added
