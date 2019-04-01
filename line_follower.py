import numpy as np
import os
import cv2
import serial
import time
VIDEO_NAME = 'LINE.mp4'
CWD_PATH = os.getcwd()

PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)

video_capture = cv2.VideoCapture(PATH_TO_VIDEO)

# arduino = serial.Serial('COM4', 9600, timeout=1)
# time.sleep(2)
# print('Connection to Arduino')
N1, N2, N3 = (0,0,0)


# video_capture.set(3, 160)
# video_capture.set(4, 120)
IM_WIDTH, IM_HEIGHT= video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) ,video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

def led_test():
    # arduino.write(("R{0}C{1}L{2}".format(N3,N2,N1)).encode())
    pass
while (True):
    # Capture the frames
    ret, frame = video_capture.read()
    # Crop the image
    # crop_img = frame[20:50, 10:160]
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Color thresholding
    ret,thresh = cv2.threshold(blur,60,255,cv2.THRESH_BINARY_INV)
    # Find the contours of the frame
    contours, hierarchy = cv2.findContours(thresh.copy(), 1, cv2.CHAIN_APPROX_NONE)
    # Find the biggest contour (if detected)
    if len(contours) > 0:
        try:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # cv2.line(frame, (cx, 0), (cx, 720), (255, 0, 0), 1)
            # cv2.line(frame, (0, cy), (1280, cy), (255, 0, 0), 1)
            # print(str(IM_WIDTH))
            cv2.line(frame, (cx, 0), (cx, int(IM_HEIGHT)), (255, 0, 0), 1)
            cv2.line(frame, (0, cy), (int(IM_WIDTH), cy), (255, 0, 0), 1)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)

            if cx >= 1280:
                print("Turn Left: {0}".format(cx))
                N3 = 1
            else:
                N3 = 0
            if cx < 1280 and cx > 640:
                print("On Track: {0}".format(cx))
                N2 = 1
            else:
                N2 = 0
            # print("{}".format(cx))
            if cx <= 640:
                print("Turn Right : {0}".format(cx))
                N1 = 1
            else:
                N1 = 0
            led_test()


            # print("CX: {0} || CY: {1}".format(cx, cy))
        except Exception as e:
            print(e)

        # cv2.line(frame, (640, 0), (640, 1080), (255, 0, 0), 1)
        # cv2.line(frame, (1280, 0), (1280, 1080), (255, 0, 0), 1)

    else:

         print ("I don't see the line")

    # Display the resulting frame

    # cv2.imshow('frame', crop_img)
    cv2.imshow('object detection', cv2.resize(frame, (800, 480)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break