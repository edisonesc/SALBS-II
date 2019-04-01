import cv2
import numpy as np
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0)
img1 = cv2.imread('signs/stop.jpg')
img2 = cv2.imread('LINE_IMG.JPG') #

orb = cv2.ORB_create()
while True:
    video = cap.read()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

    cv2.imshow('feature matching', img3)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

        break

