import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Loading the trained model
model = load_model('image_classifier.h5')

class_labels = {0: 'Good', 1: 'Moderate', 2: 'Severe' , 3:'Unhealthy' , 4:'Very_Unhealthy'}  

st.title("Air Quality Categorization")
st.write("The model categorizes pictures into 5 categories: good, moderate, severe, unhealthy, and very_unhealthy.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
st.write("It supports JPG, JPEG, and PNG formats. The color should be in RGB.")
if uploaded_file is not None:
    # Displaying the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = Image.open(uploaded_file)
    img = img.resize((256, 256)) 
    img_array = np.array(img) / 255.0  # Normalizing the image
    img_array = np.expand_dims(img_array, axis=0)  # Adding batch dimension

    # Making prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100

    # Displaying the result
    st.subheader(f"Predicted Class: {class_labels[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")
