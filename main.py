import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

# volume setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume_ctrl.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

#Media pipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

#Start Video Processing Loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    #Check for Hands and Identify Left/Right
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            lm_list = hand_landmarks.landmark
#Get Thumb and Index Finger Tips
            x1, y1 = int(lm_list[4].x * w), int(lm_list[4].y * h)  # Thumb tip
            x2, y2 = int(lm_list[8].x * w), int(lm_list[8].y * h)  # Index finger tip
#Draw Markers and Line
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

#Measure Distance Between Fingers
            dist = math.hypot(x2 - x1, y2 - y1)

#Right Hand → Volume Control
            if label == 'Right':
                vol = np.interp(dist, [20, 150], [min_vol, max_vol])
                volume_ctrl.SetMasterVolumeLevel(vol, None)
                vol_percent = int(np.interp(dist, [20, 150], [0, 100]))
                cv2.putText(img, f'Volume: {vol_percent}%', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#Left Hand → Brightness Control
            elif label == 'Left':
                bright = int(np.interp(dist, [20, 150], [0, 100]))
                try:
                    sbc.set_brightness(bright)
                except Exception as e:
                    print("Brightness error:", e)
                cv2.putText(img, f'Brightness: {bright}%', (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
#Draw Hand Landmarks
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow("Left = Brightness | Right = Volume", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
