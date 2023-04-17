import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time


cap = cv2.VideoCapture(0)
dectector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
counter = 0

folder = "Data/Like"

labels = ["Hello","I Luv u","I/Me","Like","No","Perfect","Thanks","Victory","Yes"]

while True:
  success, img = cap.read() 
  imgOutput = img.copy()
  hands, img = dectector.findHands(img)
  
  #for croping hand 
  if hands: 
    hand = hands[0]
    x, y, w, h = hand['bbox']
    
    imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
    imgCrop = img[y-offset : y+h+offset , x-offset : x+w+offset]
    
    imgCropShape = imgCrop.shape

    
    aspectRatio = h/w

    if aspectRatio >1:
      k = imgSize/h
      wCal = math.ceil(k*w)  #to roundoff approximate width
      imgResize = cv2.resize(imgCrop, (wCal , imgSize))
      imgResizeShape = imgResize.shape
      wGap = math.ceil((imgSize - wCal)/2)
      imgWhite[: , wGap:wCal+wGap] = imgResize
      prediction , index = classifier.getPrediction(imgWhite, draw=False)

    else:
      k = imgSize/w
      hCal = math.ceil(k*h)  #to roundoff approximate height
      imgResize = cv2.resize(imgCrop, (imgSize , hCal))
      imgResizeShape = imgResize.shape
      hGap = math.ceil((imgSize - hCal)/2)
      imgWhite[ hGap : hCal + hGap , :] = imgResize
      prediction , index = classifier.getPrediction(imgWhite, draw=False)
    
    
    cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+150,y-offset),(19,215,52), cv2.FILLED )

    cv2.putText(imgOutput,labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(19,215,52), 4 )

    # cv2.imshow('ImageCrop', imgCrop)
    # cv2.imshow('ImageWhite', imgWhite)

  cv2.imshow('Image', imgOutput)
  cv2.waitKey(1)

