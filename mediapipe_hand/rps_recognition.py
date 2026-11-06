
# ============================================================
# PIKIPICK - 가위바위보 인식 개선 v2
# 개선 사항:
# 1. 2D → 3D 관절 각도 계산
# 2. 손가락의 구부러짐 판별 개선
# 3. 최근 프레임을 이용한 결과 안정화
# 4. 손이 사라지면 결과 초기화
# ============================================================

import cv2
import mediapipe as mp
import time
import math

from pathlib import Path
from collections import deque, Counter


# 1. MediaPipe 설정
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"모델 없음: {MODEL_PATH}")


# 2. 손 관절 연결선
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# 손가락 이름과 관절 번호
FINGERS = {
    "INDEX":  (5, 6, 7, 8),
    "MIDDLE": (9, 10, 11, 12),
    "RING":   (13, 14, 15, 16),
    "PINKY":  (17, 18, 19, 20)
}


# 3. 3D 관절 각도 계산
def calculate_angle_3d(a, b, c):
    """
    b 관절을 중심으로 3차원 각도 계산.
    손가락이 펴질수록 180도에 가까워짐.
    """
    v1 = (
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    )

    v2 = (
        c.x - b.x,
        c.y - b.y,
        c.z - b.z
    )

    dot = sum(v1[i] * v2[i] for i in range(3))

    length1 = math.sqrt(sum(v * v for v in v1))
    length2 = math.sqrt(sum(v * v for v in v2))

    if length1 < 1e-8 or length2 < 1e-8:
        return 0.0

    cosine = dot / (length1 * length2)
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


# 4. 손가락 상태 판별
def get_finger_states(world_landmarks):
    """
    3D 각도로 네 손가락이 펴졌는지 판별.
    엄지는 가위바위보 분류에서 제외.
    """
    states = {}
    angles = {}

    for name, (mcp, pip, dip, tip) in FINGERS.items():

        # 첫 번째 마디 각도
        pip_angle = calculate_angle_3d(
            world_landmarks[mcp],
            world_landmarks[pip],
            world_landmarks[dip]
        )

        # 두 번째 마디 각도
        dip_angle = calculate_angle_3d(
            world_landmarks[pip],
            world_landmarks[dip],
            world_landmarks[tip]
        )

        # 두 각도를 함께 고려
        extended = (
            pip_angle >= 145
            and dip_angle >= 140
        )

        states[name] = extended
        angles[name] = (pip_angle, dip_angle)

    return states, angles


# 5. 가위바위보 분류
def classify_rps(states):

    index = states["INDEX"]
    middle = states["MIDDLE"]
    ring = states["RING"]
    pinky = states["PINKY"]

    # 바위
    if not any([index, middle, ring, pinky]):
        return "ROCK"

    # 가위
    if index and middle and not ring and not pinky:
        return "SCISSORS"

    # 보
    if all([index, middle, ring, pinky]):
        return "PAPER"

    # 나머지는 확정하지 않음
    return "UNKNOWN"


# 6. MediaPipe 옵션
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_buffer=MODEL_PATH.read_bytes()
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# 7. 웹캠 시작
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("카메라 연결 실패")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("PIKIPICK 가위바위보 인식 v2 시작!")
print("ROCK = 바위 / SCISSORS = 가위 / PAPER = 보")
print("Q 키로 종료")


# 8. 최근 프레임 기록
history = deque(maxlen=7)
stable_gesture = "UNKNOWN"


# 9. 실시간 인식
try:
    with HandLandmarker.create_from_options(options) as landmarker:

        start_time = time.monotonic()
        last_timestamp_ms = -1

        print("MediaPipe 모델 로딩 성공!")

        while True:

            success, frame = cap.read()

            if not success:
                break

            # 거울 모드
            frame = cv2.flip(frame, 1)

            # RGB 변환
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # 타임스탬프
            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            timestamp_ms = max(
                timestamp_ms,
                last_timestamp_ms + 1
            )

            last_timestamp_ms = timestamp_ms

            # 손 인식
            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            height, width = frame.shape[:2]

            raw_gesture = "UNKNOWN"

            if (
                result.hand_landmarks
                and result.hand_world_landmarks
            ):

                # 화면용 2D 좌표
                landmarks = result.hand_landmarks[0]

                # 분류용 3D 좌표
                world_landmarks = result.hand_world_landmarks[0]

                # 손가락 상태 분석
                states, angles = get_finger_states(
                    world_landmarks
                )

                # 가위바위보 분류
                raw_gesture = classify_rps(states)

                # 손 관절 화면 표시
                points = []

                for lm in landmarks:
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    points.append((x, y))

                for a, b in HAND_CONNECTIONS:
                    cv2.line(
                        frame,
                        points[a],
                        points[b],
                        (0, 255, 0),
                        2
                    )

                for x, y in points:
                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 0, 255),
                        -1
                    )

                # 디버깅: 손가락 상태 확인
                for i, name in enumerate(FINGERS):
                    state = "OPEN" if states[name] else "CLOSED"

                    cv2.putText(
                        frame,
                        f"{name}: {state}",
                        (15, 130 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        1
                    )

            # 10. 결과 안정화
            history.append(raw_gesture)

            counts = Counter(history)

            gesture, count = counts.most_common(1)[0]

            if count >= 5:
                stable_gesture = gesture

            # 11. 화면 상단 표시
            cv2.rectangle(
                frame,
                (0, 0),
                (width, 95),
                (30, 30, 30),
                -1
            )

            cv2.putText(
                frame,
                f"Gesture: {stable_gesture}",
                (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Current: {raw_gesture}",
                (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # 12. 카메라 화면
            cv2.imshow(
                "PIKIPICK - RPS 3D Recognition",
                frame
            )

            # Q 키 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("카메라 종료!")
