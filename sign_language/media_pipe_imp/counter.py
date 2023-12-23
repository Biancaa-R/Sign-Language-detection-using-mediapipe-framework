# Counter using media pipe
from HandTrackingModule import FindHands
import mediapipe as mp
import cv2
import os
import time

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

folderPath="sources"
folder=os.listdir(folderPath)
print(folder)

overlayList=[]
for imPath in folder:
    image=cv2.imread(f'{folderPath}\{imPath}')
    print(f'{folderPath}\{imPath}')
    overlayList.append(image)
previous=0

#detector=htm.imgDetector(detectionCon=0.7)
detector=FindHands()

while True:
    ret,img =cap.read()

    #img=htm.FindHands(img)
    #h,w,c=overlayList[0].shape
    #img[0:h,0:w]=overlayList[0]

        
    hand1_positions = detector.getPosition(img, range(21), draw=True)
    hand2_positions = detector.getPosition(img, range(21), hand_no=1, draw=True)
    for pos in hand1_positions:
        cv2.circle(img, pos, 5, (0,255,0), cv2.FILLED)
    for pos in hand2_positions:
        cv2.circle(img, pos, 5, (255,0,0), cv2.FILLED)
    #print("Index finger up:", detector.index_finger_up(img))
    #print("Middle finger up:", detector.middle_finger_up(img)(img))
    #print("Ring finger up:", detector.ring_finger_up(img) (img))
        
    if( (detector.little_finger_up(img) in ["NO HAND FOUND",])==False):
        #print("Little finger up:", detector.little_finger_up(img)(img))
        if(detector.index_finger_up(img)==True and detector.middle_finger_up(img)==False and detector.ring_finger_up(img)==False and detector.little_finger_up(img)==False):
            print("ONE")
        if(detector.index_finger_up(img)==False and detector.middle_finger_up(img)==True and detector.ring_finger_up(img) ==False and detector.little_finger_up(img)==False):
            print("ONE")
        if(detector.index_finger_up(img)==False and detector.middle_finger_up(img)==False and detector.ring_finger_up(img) ==True and detector.little_finger_up(img)==False):
            print("ONE")
        if(detector.index_finger_up(img)==False and detector.middle_finger_up(img)==False and detector.ring_finger_up(img) ==False and detector.little_finger_up(img)==True):
            print("ONE")
        if(detector.index_finger_up(img)==False and detector.middle_finger_up(img)==False and detector.ring_finger_up(img) ==False and detector.little_finger_up(img)==False):
            print("ONE")
        if(detector.index_finger_up(img)==True and detector.middle_finger_up(img)==True and detector.ring_finger_up(img) ==False and detector.little_finger_up(img)==False):
            print("TWO")
        if(detector.index_finger_up(img)==True and detector.middle_finger_up(img)==  True and detector.ring_finger_up(img) ==True and detector.little_finger_up(img)==False):
            print("THREE")
        if(detector.index_finger_up(img)==True and detector.middle_finger_up(img)==True and detector.ring_finger_up(img) ==True and detector.little_finger_up(img)==True):
            print("FOUR")

    current=time.time()
    fps=1/(current-previous)

    previous=current

    cv2.putText(img,f'FPS:{int(fps)}',(430,70),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    cv2.imshow("Image",img)
    cv2.waitKey(1)