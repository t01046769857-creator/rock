
# ============================================================
# PIKIPICK - MediaPipe 실시간 손 관절 인식
# 기능: 웹캠으로 양손을 인식하고 21개 관절을 표시
# ============================================================

# 0. 필요한 라이브러리
import cv2
import mediapipe as mp
import time
from pathlib import Path


# 1. MediaPipe Hand Landmarker 설정
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# 2. 모델 파일 경로
# 현재 Python 파일과 같은 폴더에서 모델을 찾음
MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}"
    )

print("모델 파일 확인 완료!")


# 3. 손 관절 연결 정보
# MediaPipe는 손 하나당 21개의 관절 좌표를 제공함
HAND_CONNECTIONS = [
    # 엄지
    (0, 1), (1, 2), (2, 3), (3, 4),

    # 검지
    (0, 5), (5, 6), (6, 7), (7, 8),

    # 중지
    (5, 9), (9, 10), (10, 11), (11, 12),

    # 약지
    (9, 13), (13, 14), (14, 15), (15, 16),

    # 새끼손가락
    (13, 17), (17, 18), (18, 19), (19, 20),

    # 손바닥
    (0, 17)
]


# 4. 손 인식 옵션
# 중요: 한글 경로 문제를 피하기 위해 model_asset_buffer 사용
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_buffer=MODEL_PATH.read_bytes()
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# 5. 웹캠 연결
# 0 = 기본 카메라
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError(
        "웹캠 연결 실패! 카메라 연결과 사용 권한을 확인하세요."
    )

# 카메라 화면 크기 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("웹캠 연결 성공!")
print("종료하려면 카메라 화면에서 Q를 누르세요.")


# 6. 실시간 손 인식
try:
    with HandLandmarker.create_from_options(options) as landmarker:

        print("MediaPipe 모델 로딩 성공!")

        start_time = time.monotonic()
        last_timestamp_ms = -1

        while True:

            # 7. 웹캠 프레임 읽기
            success, frame = cap.read()

            if not success:
                print("카메라 영상 읽기 실패!")
                break

            # 거울처럼 좌우 반전
            frame = cv2.flip(frame, 1)

            # 8. OpenCV BGR → MediaPipe RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # 9. VIDEO 모드용 타임스탬프
            # 프레임마다 이전 값보다 커야 함
            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            timestamp_ms = max(
                timestamp_ms,
                last_timestamp_ms + 1
            )

            last_timestamp_ms = timestamp_ms

            # 10. 손 관절 인식
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            height, width = frame.shape[:2]

            # 11. 인식된 손마다 관절 표시
            for hand_landmarks in result.hand_landmarks:

                points = []

                # 관절 좌표를 화면 픽셀로 변환
                for landmark in hand_landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    points.append((x, y))

                # 초록색 관절 연결선
                for start_idx, end_idx in HAND_CONNECTIONS:
                    cv2.line(
                        frame,
                        points[start_idx],
                        points[end_idx],
                        (0, 255, 0),
                        2
                    )

                # 빨간색 관절 점
                for x, y in points:
                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 0, 255),
                        -1
                    )

            # 12. 인식된 손 개수
            hand_count = len(result.hand_landmarks)

            cv2.putText(
                frame,
                f"Detected hands: {hand_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # 13. 웹캠 화면 출력
            cv2.imshow(
                "PIKIPICK - Hand Tracking",
                frame
            )

            # 14. Q 키로 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    # 15. 웹캠과 창 정리
    cap.release()
    cv2.destroyAllWindows()

    print("웹캠 종료 완료!")
