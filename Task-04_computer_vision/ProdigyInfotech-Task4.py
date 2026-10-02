import cv2
import mediapipe as mp
import time

from pycaw.pycaw import AudioUtilities


MODEL_PATH = "gesture_recognizer.task"


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode = mp.tasks.vision.RunningMode


options = GestureRecognizerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)


recognizer = GestureRecognizer.create_from_options(options)


device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume


connections = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (9, 10),
    (10, 11),
    (11, 12),

    (13, 14),
    (14, 15),
    (15, 16),

    (17, 18),
    (18, 19),
    (19, 20),

    (5, 9),
    (9, 13),
    (13, 17),

    (0, 17)
]


def draw_hand(frame, landmarks):

    height, width, _ = frame.shape

    points = []

    for landmark in landmarks:

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            5,
            (255, 0, 0),
            -1
        )

    for start, end in connections:

        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 0, 0),
            2
        )

    return points


def change_volume(direction):

    current = volume.GetMasterVolumeLevelScalar()

    if direction == "up":

        current = min(
            1.0,
            current + 0.05
        )

    elif direction == "down":

        current = max(
            0.0,
            current - 0.05
        )

    volume.SetMasterVolumeLevelScalar(
        current,
        None
    )


def get_action(gesture):

    actions = {

        "Thumb_Up": "VOLUME UP",

        "Thumb_Down": "VOLUME DOWN",

        "Open_Palm": "PAUSE",

        "Closed_Fist": "STOP",

        "Victory": "NEXT",

        "Pointing_Up": "SELECT",

        "ILoveYou": "CANCEL"
    }

    return actions.get(
        gesture,
        "NO ACTION"
    )


cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera could not be opened.")

    exit()


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    960
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


start_time = time.time()

last_volume_change = 0


while True:

    success, frame = cap.read()

    if not success:

        print("Unable to read camera.")

        break


    frame = cv2.flip(
        frame,
        1
    )


    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    timestamp = int(
        (time.time() - start_time) * 1000
    )


    result = recognizer.recognize_for_video(
        mp_image,
        timestamp
    )


    gesture_name = "NO HAND"
    confidence = 0
    action = "SHOW YOUR HAND"


    if result.gestures:

        if len(result.gestures[0]) > 0:

            gesture_category = result.gestures[0][0]

            gesture_name = gesture_category.category_name

            confidence = gesture_category.score * 100

            action = get_action(
                gesture_name
            )


    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        points = draw_hand(
            frame,
            landmarks
        )


        x_values = [
            p[0] for p in points
        ]

        y_values = [
            p[1] for p in points
        ]


        x1 = max(
            0,
            min(x_values) - 25
        )

        y1 = max(
            0,
            min(y_values) - 25
        )

        x2 = min(
            frame.shape[1],
            max(x_values) + 25
        )

        y2 = min(
            frame.shape[0],
            max(y_values) + 25
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        if gesture_name == "Thumb_Up":

            current_time = time.time()

            if current_time - last_volume_change > 0.6:

                change_volume("up")

                last_volume_change = current_time


        elif gesture_name == "Thumb_Down":

            current_time = time.time()

            if current_time - last_volume_change > 0.6:

                change_volume("down")

                last_volume_change = current_time


    volume_percent = int(
        volume.GetMasterVolumeLevelScalar() * 100
    )


    cv2.rectangle(
        frame,
        (10, 10),
        (430, 145),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        "Gesture: " + gesture_name,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "Confidence: {:.1f}%".format(
            confidence
        ),
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Action: " + action,
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Volume: {}%".format(
            volume_percent
        ),
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Press Q to Exit",
        (
            20,
            frame.shape[0] - 20
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Prodigy Infotech - Task 04",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


cap.release()

recognizer.close()

cv2.destroyAllWindows()