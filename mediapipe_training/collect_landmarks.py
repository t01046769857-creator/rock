
# ============================================================
# PIKIPICK - 가위바위보 학습 데이터 수집
# 목적: MediaPipe 손 관절 3D 좌표와 정답 저장
# 저장: mediapipe_training/landmarks_dataset.csv
# ============================================================

# 0. 필요한 라이브러리
import cv2
import mediapipe as mp
import csv
import time

from pathlib import Path
from datetime import datetime


# 1. 프로젝트 경로 설정
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODEL_PATH = (
    PROJECT_DIR / "mediapipe_hand" / "hand_landmarker.task"
)

CSV_PATH = BASE_DIR / "landmarks_dataset.csv"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"모델이 없습니다: {MODEL_PATH}")


# 2. 정답 키 설정
KEY_MAP = {
    ord("1"): "ROCK",
    ord("2"): "SCISSORS",
    ord("3"): "PAPER",
    ord("4"): "NONE"
}

# NONE은 손이 보이지만 가위바위보가 아닌 동작
# 예: 손가락 하나만 펴기, 엄지 올리기


# 3. CSV 컬럼 설정
# 21개 관절 * (x, y, z) = 63개 특징
FEATURE_COLUMNS = []

for i in range(21):
    FEATURE_COLUMNS += [
        f"x{i}", f"y{i}", f"z{i}"
    ]

FIELDNAMES = [
    "session_id",
    "label",
    "handedness",
    *FEATURE_COLUMNS
]


# 4. 이번 촬영 세션 ID
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# 기존 CSV에 데이터를 추가해 저장
file_exists = CSV_PATH.exists()

csv_file = CSV_PATH.open(
    "a", newline="", encoding="utf-8-sig"
)

writer = csv.DictWriter(
    csv_file,
    fieldnames=FIELDNAMES
)

if not file_exists or CSV_PATH.stat().st_size == 0:
    writer.writeheader()


# 5. MediaPipe Hand Landmarker 설정
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
Options = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = Options(
    base_options=BaseOptions(
        model_asset_buffer=MODEL_PATH.read_bytes()
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)


# 6. 카메라 연결
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    csv_file.close()
    raise RuntimeError("웹캠 연결 실패!")

counts = {
    "ROCK": 0,
    "SCISSORS": 0,
    "PAPER": 0,
    "NONE": 0
}

print("PIKIPICK 학습 데이터 수집 시작!")
print("1=바위, 2=가위, 3=보, 4=기타 동작")
print("Q=저장 후 종료")


# 7. 웹캠으로 손 관절 수집
try:
    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        start_time = time.monotonic()
        last_timestamp = -1

        while True:
            success, frame = cap.read()

            if not success:
                print("카메라 영상 읽기 실패")
                break

            # 거울 화면
            frame = cv2.flip(frame, 1)

            # MediaPipe 이미지 변환
            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # VIDEO 모드용 타임스탬프
            timestamp = int(
                (time.monotonic() - start_time) * 1000
            )

            timestamp = max(
                timestamp, last_timestamp + 1
            )
            last_timestamp = timestamp

            # 8. 손 관절 추출
            result = landmarker.detect_for_video(
                mp_image, timestamp
            )

            detected = bool(
                result.hand_world_landmarks
            )

            # 손 추적 상태 표시
            status = "HAND DETECTED" if detected else "NO HAND"

            cv2.putText(
                frame,
                status,
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if detected else (0, 0, 255),
                2
            )

            # 현재 수집 개수 표시
            for i, (label, count) in enumerate(counts.items()):
                cv2.putText(
                    frame,
                    f"{label}: {count}",
                    (15, 70 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

            cv2.imshow(
                "PIKIPICK - Collect Landmarks",
                frame
            )

            # 9. 숫자 키를 누르면 정답과 관절 좌표 기록
            key = cv2.waitKey(1) & 0xFF

            if key in KEY_MAP:
                label = KEY_MAP[key]

                # 손이 검출된 경우에만 저장
                if not detected:
                    print("손이 보이지 않아 저장하지 않았습니다.")
                    continue

                # 21개 3D 관절 정보
                landmarks = result.hand_world_landmarks[0]

                # 왼손/오른손 정보
                handedness = result.handedness[0][0].category_name

                row = {
                    "session_id": SESSION_ID,
                    "label": label,
                    "handedness": handedness
                }

                for i, lm in enumerate(landmarks):
                    row[f"x{i}"] = lm.x
                    row[f"y{i}"] = lm.y
                    row[f"z{i}"] = lm.z

                # CSV에 즉시 추가
                writer.writerow(row)
                csv_file.flush()

                counts[label] += 1

                print(
                    f"[저장] {label} / "
                    f"이번 세션 {counts[label]}개"
                )

            elif key == ord("q"):
                break

finally:
    # 10. 종료 및 파일 저장
    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

    print("수집 완료!")
    print("이번 세션 수집 개수:", counts)
    print("저장 경로:", CSV_PATH)
