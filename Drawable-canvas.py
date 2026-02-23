import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests
import numpy as np
import io

API_URL = 'http://localhost:8000/canvas/'

drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", "freedraw"
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 50, 20)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#ffffff")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)


canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=280,
    width = 280,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)


if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)


if st.button("Send for Prediction"):
    if canvas_result.image_data is not None:
        with st.spinner("Sending prediction..."):
            image_array = canvas_result.image_data.astype(np.uint8)
            pil_image = Image.fromarray(image_array)
            img_bytes = io.BytesIO()
            pil_image.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            image_sent = {'image_sent':("canvas.png",img_bytes,"image/png")}
            try:
                response = requests.post(API_URL,files=image_sent)
                response.raise_for_status()
                result = response.json()
                st.success(result["message"])
                st.write(f"Prediction: {result.get('predicted_number', 'No prediction found')} (Prediction: {result.get('probabilities', 'No probabilities found'):.2f}%)")

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to API: {e}")
    else:
        st.warning("Please draw something on the canvas first!")