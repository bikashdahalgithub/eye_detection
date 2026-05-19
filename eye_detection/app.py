import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model(
    "eye_detection_model.keras"
)

# App title
st.title(" Open/Closed Eye Detection")

st.write(
    "Upload an eye image to predict whether the eye is open or closed."
)

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(image, caption="Uploaded Image")

    # Resize image
    image = image.resize((64,64))

    # Convert image to array
    img_array = np.array(image)

    # Normalize
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    # Result
    if prediction[0][0] > 0.5:
        st.success(" Open Eye")
    else:
        st.error("Closed Eye")