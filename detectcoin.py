import os
import cv2 
import numpy as np

def detection(file_name, window_name):
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray_blurred = cv2.blur(gray, (2, 2)) 
    th3 = cv2.adaptiveThreshold(gray_blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,20)
    kernel = np.ones((4,4),np.uint8)
    opening = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)

    edged = cv2.Canny(th3, 30, 200)

    img_dilation = cv2.dilate(th3, kernel, iterations=1)
    erode = cv2.erode(img_dilation, kernel, iterations=2) 

    cnts = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = 255 - original[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        break
    
    cv2.imshow('edged', edged)

    cv2.imshow('th3', th3)
    cv2.imshow('erode', erode)
    cv2.imshow('img_dilation', img_dilation)
    cv2.imshow('opening', opening)

    detected_circles = cv2.HoughCircles(opening, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 50, maxRadius = 110) 
    if detected_circles is not None: 
        detected_circles = np.uint16(np.around(detected_circles)) 
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
            cv2.circle(img, (a, b), r, (255, 255, 255), -1)
            cv2.imshow(window_name, img) 
        cv2.waitKey(0) 


# Path = "Coin/square/"
path = r'D:/Folder/Kuliah/KK  E Smt. 5/DR_FPKKE/nimg/1.jpg'
nama_window = "tes"
detection(path, nama_window)
# nama_window = "tes"
# for dirName, subdirList, fileList in os.walk(path):
#     for filename in fileList:
#         if filename[-4:]!='.jpg':
#             continue
#         sesuatu = os.path.join(dirName, filename)
#         detection(sesuatu, nama_window+filename)