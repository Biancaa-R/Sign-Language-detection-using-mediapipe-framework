import cv2
import mediapipe as mp

mp_draw=mp.solutions.drawing_utils
mp_hand= mp.solutions.hands
video=cv2.VideoCapture(0)

with mp_hand.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5 ) as hands:

    while True:
        ret,image=video.read()
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        results=hands.process(image)
        image.flags.writeable=True
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image,hand_landmark,mp_hand.HAND_CONNECTIONS)
        cv2.imshow("frame",image)
        k=cv2.waitKey(1)
        if k==ord('q'):
            break