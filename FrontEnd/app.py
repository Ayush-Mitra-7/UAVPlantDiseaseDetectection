import os
import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO


def save_uploaded_image(upload_dir):
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image.save(os.path.join(upload_dir, uploaded_file.name))
        return image, uploaded_file.name
    return None, None


def predict(image_path):
    model = YOLO(r"best.pt")
    result_img = model.predict(source=image_path, imgsz=512)
    res_plotted = result_img[0].plot()
    return res_plotted


upload_dir = "test_img_web"
comp_dir = "test_img_web_comp"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)
if not os.path.exists(comp_dir):
    os.makedirs(comp_dir)
st.set_page_config(page_title="Plant Detector", layout="wide")
st.title("Plant Disease Detector Using UAV Image")

image, filename = save_uploaded_image(upload_dir)

if image is not None:
    result_image = predict(os.path.join(upload_dir, filename))
    col1, col2 = st.columns(2)
    col1.subheader("Original Image")
    col1.image(image, use_column_width=True)

    col2.subheader("Predicted Image")
    col2.image(result_image, use_column_width=True)
    cv2.imwrite(os.path.join(comp_dir, filename), result_image[:, :, ::-1])
    st.success("Prediction complete.")
