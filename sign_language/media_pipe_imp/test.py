import cv2
from HandTrackingModule import FindHands

cap = cv2.VideoCapture(0)
detector = FindHands()


while True:
    succeed, img = cap.read()
    hand1_positions = detector.getPosition(img, range(21), draw=False)
    hand2_positions = detector.getPosition(img, range(21), hand_no=1, draw=False)
    for pos in hand1_positions:
        cv2.circle(img, pos, 5, (0,255,0), cv2.FILLED)
    for pos in hand2_positions:
        cv2.circle(img, pos, 5, (255,0,0), cv2.FILLED)
    print("Index finger up:", detector.index_finger_up(img))
    print("Middle finger up:", detector.middle_finger_up(img))
    print("Ring finger up:", detector.ring_finger_up(img))
    print("Little finger up:", detector.little_finger_up(img))
    cv2.imshow("Image", img)
    if cv2.waitKey(10) == ord('q'):
        break