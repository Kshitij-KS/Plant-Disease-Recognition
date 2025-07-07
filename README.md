# Plant Disease Detection System

## Overview
The Plant Disease Detection System is a web application designed to help farmers and agricultural enthusiasts identify plant diseases quickly and accurately. Utilizing a Convolutional Neural Network (CNN) model trained on a diverse dataset of plant images, this application provides predictions based on user-uploaded images. The model achieved an impressive accuracy of **90.97%**, demonstrating its effectiveness in classifying various plant diseases.

## How It Works
1. **Image Upload**: Users can upload images of plants in JPEG or PNG format through the web interface.
2. **Image Preprocessing**: The uploaded image is resized and normalized to meet the input requirements of the CNN model.
3. **Model Prediction**: The preprocessed image is fed into the trained CNN model, which outputs the predicted class label for the plant disease.
4. **Result Display**: The application displays the predicted disease along with a confidence score, helping users understand the likelihood of the prediction.

## Code Structure
The project is organized into several key components:

- **Frontend.py**: The main file that runs the Streamlit web application. It handles user interactions, image uploads, and displays predictions.
- **requirements.txt**: A file listing all necessary Python packages required to run the application.

## Datasets
The model was trained using a dataset sourced from Kaggle. You can find the dataset [here](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data).

### Model Performance
The CNN model achieved an accuracy of **91.56%** on the validation set, indicating its high effectiveness in classifying plant diseases.

## Installation

### Prerequisites
Ensure you have Python 3.7 or higher installed on your machine.


## Technologies Used
- **Python**: The programming language used for the application.
- **Streamlit**: A framework for building web applications quickly.
- **TensorFlow**: An open-source library for machine learning used to build the CNN model.
- **OpenCV**: A library for image processing.
- **NumPy**: A library for numerical computations in Python.
- **Pillow (PIL)**: A library for image handling.

## Installation

### Prerequisites
Make sure you have Python 3.7 or higher installed on your machine. You will also need to install the required libraries.

### Step 1: Clone the Repository
git clone https://github.com/Kshitij-KS/Plant-Disease-Detection-System.git
cd Plant-Disease-Detection-System

### Step 2: Install Required Packages
Install the necessary packages using pip:
pip install -r requirements.txt

### Step 3: Run the Application
To start the Streamlit application, run the following command:
streamlit run app.py

## Usage
1. Navigate to the application in your web browser (usually at `http://localhost:8501`).
2. Click on Disease Recognition Button.
3. Upload an image of a plant.
4. Click "Show Image" to view the uploaded image.
5. Click "Predict" to get the disease prediction.

## Model Information
The model used in this application is a Convolutional Neural Network (CNN) trained on a dataset of various plant diseases. The model is capable of identifying multiple diseases across different types of plants.

## Contributing
Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request.


