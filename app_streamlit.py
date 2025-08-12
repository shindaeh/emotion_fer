
# app_streamlit.py     # Streamlit 데모(파일/웹캠)


import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from utils_face import detect_faces_bgr

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="😊")

MODEL_PATH = "emotion_mobilenetv2.keras"
IMG_SIZE = (224, 224)
CLASSES = ["angry","disgust","fear","happy","sad","surprise","neutral"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def predict_face_rgb(rgb):
    arr = cv2.resize(rgb, IMG_SIZE).astype(np.float32)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)
    prob = model.predict(arr, verbose=0)[0]
    idx = int(prob.argmax())
    return CLASSES[idx], float(prob[idx]), prob

def draw_boxes(bgr, boxes, labels):
    for (x,y,w,h,lab,conf) in labels:
        cv2.rectangle(bgr, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(bgr, f"{lab} {conf:.2f}", (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return bgr

st.title("😊 Facial Emotion Recognition (MobileNetV2)")
st.caption("파일 업로드 또는 웹캠 촬영으로 감정을 분류합니다.")

model = load_model()
tab1, tab2 = st.tabs(["📁 파일 업로드", "📸 웹캠 촬영"])

with tab1:
    up = st.file_uploader("이미지를 업로드하세요 (jpg/png)", type=["jpg","jpeg","png"])
    if up is not None:
        pil = Image.open(up).convert("RGB")
        rgb = np.array(pil)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        boxes = detect_faces_bgr(bgr)

        labels = []
        if len(boxes)==0:
            lab, conf, prob = predict_face_rgb(rgb)
            st.write(f"예측: **{lab}** (conf={conf:.2f})")
            st.bar_chart({c: p for c,p in zip(CLASSES, prob)})
            st.image(pil, caption="입력 이미지", use_container_width=True)
        else:
            for (x,y,w,h) in boxes:
                face_rgb = rgb[y:y+h, x:x+w]
                lab, conf, prob = predict_face_rgb(face_rgb)
                labels.append((x,y,w,h,lab,conf))
            out = draw_boxes(bgr.copy(), labels)
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                     caption="검출/예측 결과", use_container_width=True)

with tab2:
    pic = st.camera_input("웹캠으로 촬영하세요")
    if pic is not None:
        pil = Image.open(io.BytesIO(pic.getvalue())).convert("RGB")
        rgb = np.array(pil)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        boxes = detect_faces_bgr(bgr)

        labels = []
        if len(boxes)==0:
            lab, conf, prob = predict_face_rgb(rgb)
            st.write(f"예측: **{lab}** (conf={conf:.2f})")
            st.bar_chart({c: p for c,p in zip(CLASSES, prob)})
            st.image(pil, caption="입력 이미지", use_container_width=True)
        else:
            for (x,y,w,h) in boxes:
                face_rgb = rgb[y:y+h, x:x+w]
                lab, conf, prob = predict_face_rgb(face_rgb)
                labels.append((x,y,w,h,lab,conf))
            out = draw_boxes(bgr.copy(), labels)
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                     caption="검출/예측 결과", use_container_width=True)

