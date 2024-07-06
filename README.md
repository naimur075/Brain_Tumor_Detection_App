# Brain_Tumor_Detection_App
Welcome to the Brain Tumor Detection project! This repository contains the code and resources for developing a convolutional neural network (CNN) model aimed at detecting brain tumors from MRI scans. The goal of this project is to leverage deep learning techniques to provide an accurate and efficient tool for assisting medical professionals in diagnosing brain tumors.
## Features

- **Deep Learning Models**: Implementation of various CNN models from scratch as well as using popular deep learning libraries.
- **Streamlit Interface**: An interactive web application built with Streamlit to upload MRI scans and view predictions.
- **Multi-class Extension**: Capability to extend the classifier to handle additional types of medical images, such as chest X-rays and hand X-rays.
- **Performance Metrics**: Comprehensive evaluation of model performance using metrics such as accuracy, precision, recall, and F1 score.

## Description
This repository is organized into several files and folders, each serving a specific purpose in the brain tumor detection project. Below is a detailed description of each component:

- **[Brain Tumor Detection.ipynb](Brain%20Tumor%20Detection.ipynb)**: This file contains the build, train, and test processes of a brain tumor detection model from scratch using NumPy and scikit-learn.
- **[Data Augmentation.ipynb](Data%20Augmentation.ipynb)**: This file contains data augmentation code to enhance the dataset.
- **[best_model_f1_0.7660_epoch_99.h5](best_model_f1_0.7660_epoch_99.h5)**: This is the best-trained model with an accuracy of 76.60%, saved in h5 format.
- **[requirements.txt](requirements.txt)**: This file contains all the necessary and required libraries for this project.
- **[augmented data](augmented%20data)**: This folder contains a dataset of 2000 images equally divided into 'yes' and 'no' subfolders.
- **[Real Time Model](Real%20Time%20Model)**: This is an empty folder which will be needed during web app integration in Streamlit.
- **[Web_Apps](Web_Apps)**: This folder contains four files:
  - **[App.py](Web_Apps/App.py)**: This is the web app in Python using Streamlit. It provides an interface for users on a local host to test the trained model `best_model_f1_0.7660_epoch_99.h5`.
  - **[App2.py](Web_Apps/App2.py)**: This is an update to `App.py` which enables **real-time training and testing** on a local host.
  - **[App3.py](Web_Apps/App3.py)**: This is an update to `App2.py` which enables **new training and existing training** along with real-time training and testing on a local host.
  - **[App8.py](Web_Apps/App8.py)**: This is an update to all previous versions, enabling real-time training with both **Scratch Model (NumPy & scikit-learn)** and **Library Model (TensorFlow & Keras)**. Additionally, it displays ROC curves and confusion matrices for the trained models.

To run these apps, you must install Streamlit on your PC. Once installed, navigate to the directory containing the app and run it using the following command:

```bash
python -m streamlit run app.py
