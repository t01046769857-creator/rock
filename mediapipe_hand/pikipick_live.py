
# ============================================================
# PIKIPICK - 실시간 가위바위보 AI 손동작 인식
#
# 1. MediaPipe: 손 관절 21개 검출
# 2. 관절 각도: 검지 / 중지 / 약지 / 새끼
# 3. RandomForest: ROCK / SCISSORS / PAPER 예측
# 4. OpenCV: 웹캠 화면에 결과 표시
#
# 종료: Q
# ============================================================

# 0. 라이브러리
import cv2
import time
import joblib
import numpy as np
import mediapipe as mp

from pathlib import Path
from collections import deque, Counter


# 1. 프로젝트 파일 경로
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"

AI_MODEL_PATH = (
    PROJECT_DIR
    / "mediapipe_training"
    / "pikipick_angle_randomforest.joblib"
)

# 파일 존재 여부 확인
if not HAND_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"MediaPipe 모델 없음: {HAND_MODEL_PATH}"
    )

if not AI_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"학습된 AI 모델 없음: {AI_MODEL_PATH}"
    )


# 2. 학습된 RandomForest 모델 불러오기
# 직접 학습해서 저장한 신뢰할 수 있는 파일만 로드
model_package = joblib.load(AI_MODEL_PATH)

model = model_package["model"]

print("[PIKIPICK] 모델 불러오기 성공!")
print("분류 클래스:", model.classes_)


# 3. 관절 사이 각도 계산
# 학습할 때 사용한 calculate_angle과 동일한 공식
def calculate_angle(a, b, c):

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
    )

    if denominator < 1e-8:
        return np.nan

    cos_angle = np.dot(ba, bc) / denominator
    cos_angle = np.clip(cos_angle, -1, 1)

    return np.degrees(np.arccos(cos_angle))


# 4. 손 관절에서 네 손가락 각도 추출
# 학습에서 사용한 관절 인덱스와 순서 유지
FINGER_JOINTS = {
    "INDEX": (5, 6, 7),
    "MIDDLE": (9, 10, 11),
    "RING": (13, 14, 15),
    "PINKY": (17, 18, 19)
}

ANGLE_COLUMNS = [
    "INDEX", "MIDDLE", "RING", "PINKY"
]


def extract_finger_angles(landmarks):

    # MediaPipe 3D 손 관절 좌표
    points = np.array([
        [lm.x, lm.y, lm.z]
        for lm in landmarks
    ])

    angles = {}

    for finger, (a, b, c) in FINGER_JOINTS.items():

        angles[finger] = calculate_angle(
            points[a],
            points[b],
            points[c]
        )

    # 학습 당시와 똑같은 특징 순서
    return np.array([
        angles[name]
        for name in ANGLE_COLUMNS
    ], dtype=float).reshape(1, -1)


# 5. MediaPipe 설정
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        # 한글 Windows 경로에서도 안전하게 로드
        model_asset_buffer=HAND_MODEL_PATH.read_bytes()
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)


# 6. 웹캠 실행
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없습니다.")


# 7. 예측 결과 안정화
# 최근 5프레임의 다수결
prediction_history = deque(maxlen=5)

start_time = time.monotonic()
last_timestamp = -1

print("\nPIKIPICK 실시간 손동작 인식 시작!")
print("손으로 ROCK / SCISSORS / PAPER를 보여주세요.")
print("Q 키를 누르면 종료합니다.")


# 8. 실시간 인식
try:
    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            success, frame = cap.read()

            if not success:
                print("카메라 프레임을 읽지 못했습니다.")
                break

            # 거울 화면
            frame = cv2.flip(frame, 1)

            # OpenCV BGR -> RGB
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # MediaPipe 이미지 생성
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # VIDEO 모드 타임스탬프
            timestamp = int(
                (time.monotonic() - start_time) * 1000
            )
            timestamp = max(
                timestamp,
                last_timestamp + 1
            )
            last_timestamp = timestamp

            # 손 관절 검출
            result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            gesture = "NO HAND"
            score = 0.0

            if result.hand_world_landmarks:

                # 검출된 손의 3D 관절 좌표
                landmarks = result.hand_world_landmarks[0]

                # 네 손가락 관절 각도 계산
                features = extract_finger_angles(
                    landmarks
                )

                # 유효한 관절 각도인지 확인
                if np.isfinite(features).all():

                    # 모델 예측
                    prediction = model.predict(features)[0]

                    # 클래스별 예측 확률
                    probabilities = model.predict_proba(
                        features
                    )[0]

                    score = float(
                        probabilities.max()
                    )

                    # 최근 예측 결과 저장
                    prediction_history.append(prediction)

                    # 최근 5프레임 다수결
                    gesture = Counter(
                        prediction_history
                    ).most_common(1)[0][0]

                else:
                    prediction_history.clear()
                    gesture = "INVALID HAND"

            else:
                # 손이 사라지면 이전 결과 초기화
                prediction_history.clear()

            # 9. 웹캠 결과 표시
            cv2.putText(
                frame,
                f"PIKIPICK: {gesture}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            if gesture not in ("NO HAND", "INVALID HAND"):
                cv2.putText(
                    frame,
                    f"Frame max probability: {score:.1%}",
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            cv2.putText(
                frame,
                "Q: QUIT",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "PIKIPICK - Real-time RPS AI",
                frame
            )

            # 종료 키
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

finally:
    # 10. 카메라 및 창 정리
    cap.release()
    cv2.destroyAllWindows()

    print("\nPIKIPICK 종료!")
