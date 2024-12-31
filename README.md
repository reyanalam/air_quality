# Air Quality Categorization

![bandicam2024-12-1522-02-49-142-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/af21d56b-70de-4625-a1b0-be270c6e10fe)

## Project Overview

The **AeroVision AI** project leverages deep learning techniques to classify images of air quality into five distinct categories: **Good**, **Moderate**, **Severe**, **Unhealthy**, and **Very Unhealthy**. 
The model has been trained on a dataset consisting of 5000 labeled images, each representing different air quality conditions. This classification can aid in understanding and monitoring environmental conditions in real-time.

### Usefulness of the Model:
This model is useful for real-time air quality monitoring through visual recognition. By categorizing images into different air quality classes, it enables individuals, businesses, and organizations to gain immediate insight into their environment. This can be particularly beneficial in areas where air quality monitoring infrastructure is lacking, or where rapid assessments are necessary. With applications in public health, environmental monitoring, and climate research, this model can help raise awareness and promote actions towards cleaner air.

## Objective

The goal of this project is to create an image classification model capable of identifying and categorizing air quality images into one of the following five categories:

- **Good**
- **Moderate**
- **Severe**
- **Unhealthy**
- **Very Unhealthy**

These categories correspond to different levels of air pollution and provide insight into the air quality conditions captured in the images.

## Dataset

The dataset contains around 10,000 images, each representing a different air quality condition. The images have been manually labeled into one of the five classes mentioned above. These images are used to train, validate, and test the deep learning model.

## Model Architecture

The deep learning model used in this project is based on a **Convolutional Neural Network (CNN)**, which is well-suited for image classification tasks. The architecture includes several layers made by using tensorflow and keras , designed to extract features from the images and progressively make predictions about the air quality class.

### Key Features:
- **Regularization:** Techniques like dropout and batch normalization were incorporated to prevent overfitting and enhance model performance.
- **Learning Rate:** The model was tested on different learning rate out of which best was choosen.
- **Evaluation:** The model's performance has been evaluated using Accuracy.

## Web Application:

In addition to the deep learning model, a Streamlit web application has been developed to provide a user-friendly interface for classifying air quality images.

### Key Features:
- **User Uploads Image:** Users can upload an image from their local device.
- **Instant Prediction:** Once the image is uploaded, the model classifies the image into one of the five air quality categories and displays the result on the web page.

## Installation and Setup

To run this project on your local machine, follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/air-quality-categorization.git

2. Navigate to the project directory:
    ```bash
    cd Image_classification

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt

4. Run the command:
    ```bash
    streamlit run app.py

## Contact

1. **Linkedin:** https://www.linkedin.com/in/reyanalam/
2. **Gmail:** reyanalam115@gmail.com
