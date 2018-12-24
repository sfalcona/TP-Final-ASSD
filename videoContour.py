import numpy as np
import cv2
from PIL import ImageGrab


#  Prueba sublime - git
cap = cv2.VideoCapture(0)

cont = 0
maxArea = 300
prevC = 0
danger = False
bgs = cv2.createBackgroundSubtractorMOG2(10,80,True)
while(True):
    
    ret, frame = cap.read()    
    frame2 = bgs.apply(frame)
    blur = cv2.GaussianBlur(frame2,(5,5),1) 
    ret, thresh_img = cv2.threshold(blur,91,255,cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    dilated = cv2.dilate(thresh_img,kernel,iterations = 6) 
    gradient = cv2.morphologyEx(thresh_img, cv2.MORPH_GRADIENT, kernel)
    opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
    contours =  cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    font = cv2.FONT_HERSHEY_COMPLEX
    color = (0,0,255) if ((prevC > 2) and danger ) else (0,255,0)
    danger = False
    cv2.putText(frame,'.',(10,10), font, 4,color,5,cv2.LINE_AA)
    
    for c in contours:
        if cv2.contourArea(c) > maxArea:
            if prevC != 0:
                danger = True
                cv2.drawContours(frame, [c], -1, (0,255,0), 3)
#                 [x,y,w,h] = cv2.boundingRect(c)
#                 cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,255),2)
        prevC = len(c)
   


    cv2.imshow('frame',frame) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()