import numpy as np
import cv2

cap = cv2.VideoCapture('C:\\Users\\Radhu\\Videos\\Youtube Videos\\3 Idiots - Official Trailer.mp4')

face_cascade = cv2.CascadeClassifier('C:\\Python27\\Assignments\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Python27\\Assignments\\haarcascade_eye.xml')

i = 0
j = 0

while(cap.isOpened()):

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    cv2.imshow('frame',frame)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
           j = j + 1
           i = i + 1
           #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
           #roi_gray = gray[y:y+h, x:x+w]
           #roi_color = frame[y:y+h, x:x+w]
           #eyes = eye_cascade.detectMultiScale(roi_gray)
           #for (ex,ey,ew,eh) in eyes:
           #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
               
    if(j > 0):
        j = 0
        cv2.imwrite('C:\\Users\\Radhu\\Documents\\Projects\\FaceDetection\\face' + str(i)+'.jpg',frame)
#        cv2.imshow('face' + str(i),frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
        
