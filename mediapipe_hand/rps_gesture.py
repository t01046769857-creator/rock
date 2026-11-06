
# ============================================================
# PIKIPICK - MediaPipe 가위바위보 진단 버전
#
# 기능
# 1. 웹캠 실시간 손 인식
# 2. 손 관절 21개 표시
# 3. 가위 / 바위 / 보 분류
# 4. 손 추적 성공 여부 확인
# 5. 모델 분류 결과 및 신뢰도 확인
# 6. 최근 프레임 기반 결과 안정화
# ============================================================

# 0. 라이브러리
import cv2
import mediapipe as mp
import time

from pathlib import Path
from collections import deque, Counter


# 1. MediaPipe 모델 설정
BaseOptions = mp.tasks.BaseOptions

GestureRecognizer = (
    mp.tasks.vision.GestureRecognizer
)

GestureRecognizerOptions = (
    mp.tasks.vision.GestureRecognizerOptions
)

RunningMode = mp.tasks.vision.RunningMode


# 2. 모델 경로
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "gesture_recognizer.task"
)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"모델 파일이 없습니다: {MODEL_PATH}"
    )

print("모델 파일 확인 완료!")


# 3. 가위바위보 분류 기준
RPS_MAP = {
    "Closed_Fist": "ROCK",
    "Victory": "SCISSORS",
    "Open_Palm": "PAPER"
}


# 4. 손 관절 연결선
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


# 5. MediaPipe 옵션
# 한글 폴더 경로 문제를 피하기 위해 buffer 사용
options = GestureRecognizerOptions(
    base_options=BaseOptions(
        model_asset_buffer=MODEL_PATH.read_bytes()
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# 6. 웹캠 연결
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError(
        "웹캠 연결 실패! 카메라를 확인하세요."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("웹캠 연결 성공!")
print("PIKIPICK 가위바위보 진단 시작!")
print("가위 자세를 2~3초 유지해 보세요.")
print("종료하려면 카메라 창에서 Q를 누르세요.")


# 7. 결과 안정화
history = deque(maxlen=7)

stable_gesture = "UNKNOWN"

# 터미널 출력 간격: 0.5초
last_log_time = 0


# 8. 실시간 인식
try:
    with GestureRecognizer.create_from_options(
        options
    ) as recognizer:

        print("MediaPipe 모델 로딩 성공!")

        start_time = time.monotonic()
        last_timestamp_ms = -1

        while True:

            # 9. 카메라 프레임 읽기
            success, frame = cap.read()

            if not success:
                print("카메라 영상 읽기 실패!")
                break

            # 거울처럼 좌우 반전
            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            # 10. BGR -> RGB 변환
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # 11. VIDEO 모드 타임스탬프
            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            timestamp_ms = max(
                timestamp_ms,
                last_timestamp_ms + 1
            )

            last_timestamp_ms = timestamp_ms

            # 12. MediaPipe 동작 인식
            result = recognizer.recognize_for_video(
                mp_image,
                timestamp_ms
            )

            # 13. 손 추적 여부 확인
            hand_detected = (
                len(result.hand_landmarks) > 0
            )

            raw_name = "None"
            confidence = 0.0
            raw_gesture = "UNKNOWN"

            # 14. 분류 모델 결과 확인
            if result.gestures and result.gestures[0]:

                category = result.gestures[0][0]

                raw_name = category.category_name
                confidence = category.score

                raw_gesture = RPS_MAP.get(
                    raw_name,
                    "UNKNOWN"
                )

            # 15. 결과 안정화
            # 손이 사라지면 이전 결과를 즉시 초기화
            if not hand_detected:
                history.clear()
                stable_gesture = "UNKNOWN"

            else:
                history.append(raw_gesture)

                # 최근 7프레임 중 5개 이상 일치
                if len(history) >= 5:

                    common, count = Counter(
                        history
                    ).most_common(1)[0]

                    if count >= 5:
                        stable_gesture = common
                    else:
                        stable_gesture = "UNKNOWN"

            # 16. 손 관절 표시
            for hand_landmarks in result.hand_landmarks:

                points = []

                for lm in hand_landmarks:

                    x = int(lm.x * width)
                    y = int(lm.y * height)

                    points.append((x, y))

                # 초록색 관절 연결선
                for a, b in HAND_CONNECTIONS:

                    cv2.line(
                        frame,
                        points[a],
                        points[b],
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

            # 17. 화면 상단 배경
            cv2.rectangle(
                frame,
                (0, 0),
                (width, 155),
                (30, 30, 30),
                -1
            )

            # 손 추적 상태
            tracking_status = (
                "SUCCESS"
                if hand_detected
                else "FAILED"
            )

            cv2.putText(
                frame,
                f"Hand Tracking: {tracking_status}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if hand_detected
                else (0, 0, 255),
                2
            )

            # 모델 실제 결과
            cv2.putText(
                frame,
                f"Model: {raw_name}",
                (15, 63),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # 신뢰도
            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}",
                (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # 최종 안정화 결과
            cv2.putText(
                frame,
                f"RPS: {stable_gesture}",
                (15, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 255),
                2
            )

            # 18. 터미널 진단 로그
            # 매 프레임 출력하면 너무 많으므로 0.5초마다 표시
            current_time = time.monotonic()

            if current_time - last_log_time >= 0.5:

                status = (
                    "손 추적 성공"
                    if hand_detected
                    else "손 추적 실패"
                )

                print(
                    f"{status} | "
                    f"Model: {raw_name} | "
                    f"Confidence: {confidence:.2f} | "
                    f"RPS: {raw_gesture}",
                    flush=True
                )

                last_log_time = current_time

            # 19. 카메라 화면 출력
            cv2.imshow(
                "PIKIPICK - RPS Diagnostic",
                frame
            )

            # Q 키로 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    # 20. 종료 처리
    cap.release()
    cv2.destroyAllWindows()

    print("카메라 종료 완료!")
