# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:00:18 2018

@author: user
"""

import cv2
import os
import time
import numpy as np
import matplotlib.pyplot as pltde

face_detector = cv2.CascadeClassifier('C:/Users/user/Desktop/TCS ML/Lecture 9/haarcascades/haarcascade_frontalface_default.xml')
count = 0
name = input("Enter your name:\n")
newpath = "C:/Users/user/Desktop/PyPics/"
appendedpath = newpath +name
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not os.path.isdir(appendedpath):
    os.mkdir(appendedpath)
while ret:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imwrite(os.path.join(appendedpath, str(count)+'.jpg'), frame[y:y+h,x:x+w])
        count += 1
        time.sleep(1)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(5)&0xFF
    if k == 27 or count > 20:
        break
cap.release()
cv2.destroyAllWindows()