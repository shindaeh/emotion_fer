
# infer_image.py   단일 이미지 추출(+얼굴검출)

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from utils_face import detect_faces_bgr

MODEL_PATH = "emotion_mobilenetv2.keras"
IMG_SIZE = (224, 224)
CLASSES = ["angry","disgust","fear","happy","sad","surprise","neutral"]

model = tf.keras.models.load_model(MODEL_PATH)

def predict_bgr_face(img_bgr):
    # RGB 변환 및 전처리
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    arr = cv2.resize(img_rgb, IMG_SIZE).astype(np.float32)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)
    prob = model.predict(arr, verbose=0)[0]
    idx = prob.argmax()
    return CLASSES[idx], float(prob[idx])

def run(image_path, out_path="result.jpg"):
    bgr = cv2.imread(image_path)
    boxes = detect_faces_bgr(bgr)
    if len(boxes)==0:
        # 얼굴 못찾으면 전체 이미지로 예측
        label, conf = predict_bgr_face(bgr)
        cv2.putText(bgr, f"{label} {conf:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        for (x,y,w,h) in boxes:
            face = bgr[y:y+h, x:x+w]
            label, conf = predict_bgr_face(face)
            cv2.rectangle(bgr, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(bgr, f"{label} {conf:.2f}", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imwrite(out_path, bgr)
    print("Saved:", out_path)

if __name__ == "__main__":
    import sys
    run(sys.argv[1], sys.argv[2] if len(sys.argv)>2 else "result.jpg")
