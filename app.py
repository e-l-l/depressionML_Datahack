import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import cv2
import joblib
import tensorflow as tf
import streamlit as st
from PIL import Image
from io import BytesIO
import base64

st.title("Damage Detection")

print(tf.__version__)
model = tf.keras.models.load_model('model.h5')

IMG_SIZE = 300
CATEGORIES = ['01-whole','00-damage']

def create_sample(x):
    for category in CATEGORIES:
        class_num = CATEGORIES.index(category)
        print(x)
        img_array = cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_COLOR)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
        return new_array

from matplotlib.image import imread
upload = st.file_uploader("Upload an image", type=("jpg","jpeg", "png"))
if upload is not None:
    image_bytes = upload.read()
    image = Image.open(BytesIO(image_bytes))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    # buffered = BytesIO()
    # image.save(buffered, format="JPEG")
    # img_str = base64.b64encode(buffered.getvalue()).decode()
    test_x=create_sample(image_bytes)
    # xpath=imread(image)
    # cv2.imshow(xpath)
    test_x = np.array(test_x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    test_x = test_x/255.0
    pred=model.predict(test_x)
    print(pred)
    if pred.argmax(axis=-1)==1:
        st.write("not damaged")
    else:
        st.write("damaged")

