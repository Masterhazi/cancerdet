import streamlit as st
import numpy as np
import cv2
from PIL import Image

# import model builder
from load_model import build_model

# load model
model = build_model()
model.load_weights("model.weights.h5")

st.title("Skin Cancer Detection")

def remove_hair(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    result = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
    return result

def preprocess(image):
    image = cv2.resize(image, (128,128))
    image = remove_hair(image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

uploaded = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded)
    image = np.array(image)

    st.image(image, caption="Uploaded Image")

    processed = preprocess(image)

    prediction = model.predict(processed)[0][0]

    st.write("Prediction score:", prediction)

    if prediction > 0.5:
        st.error("⚠️ Possible Cancer")
    else:
        st.success("✅ Likely Benign")
        