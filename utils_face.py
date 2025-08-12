
# utils_face.py      얼굴 검출 유틸(OpenCV)

import os
from typing import List, Tuple
import cv2

# OpenCV가 설치한 내부 경로에서 기본 Haar Cascade를 사용합니다.
# 배포(예: Streamlit Cloud) 환경에서도 opencv-python만 설치되어 있으면 자동으로 동작합니다.
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# 로드 가능 여부 체크
if not os.path.exists(CASCADE_PATH):
    raise RuntimeError(
        f"Haar cascade not found at {CASCADE_PATH}. "
        "Please ensure opencv-python is installed, or provide a valid cascade path."
    )

# 전역에서 한 번만 로드 (반복 생성 비용 줄임)
_FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)
if _FACE_CASCADE.empty():
    raise RuntimeError(
        "Failed to load haarcascade_frontalface_default.xml. "
        "Your OpenCV installation may be incomplete or corrupted."
    )

def detect_faces_bgr(
    bgr_image,
    scaleFactor: float = 1.1,
    minNeighbors: int = 5,
    minSize: Tuple[int, int] = (30, 30)
) -> List[Tuple[int, int, int, int]]:
    """
    BGR 이미지에서 얼굴을 검출하여 (x, y, w, h) 박스 리스트를 반환합니다.

    Parameters
    ----------
    bgr_image : np.ndarray
        OpenCV BGR 이미지
    scaleFactor : float
        이미지 피라미드 스케일 (기본 1.1)
    minNeighbors : int
        인접 검출 최소 개수 (값이 클수록 검출은 덜 민감, 오탐 감소)
    minSize : (int, int)
        최소 얼굴 크기

    Returns
    -------
    List[Tuple[int,int,int,int]]
        얼굴 바운딩 박스들의 리스트 (x, y, w, h)
    """
    if bgr_image is None or bgr_image.size == 0:
        return []

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    boxes = _FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize
    )
    return list(map(tuple, boxes))
