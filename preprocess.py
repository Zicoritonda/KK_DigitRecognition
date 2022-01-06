import cv2
import numpy as np

path = r'D:/Folder/Kuliah/KK  E Smt. 5/DR_FPKKE/nimg/1.jpg'
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

alpha = 1.0 # Simple contrast control
beta = -1.0    # Simple brightness control
nimg = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

cv2.imshow('a',gray)
cv2.imshow('b',nimg)
cv2.imshow('c',thresh1)

cv2.waitKey(0)
cv2.destroyAllWindows()