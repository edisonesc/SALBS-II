import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from PIL import Image
import serial
import time
import pyzbar.pyzbar as pyzbar
import main_helper as mh
import traceback

import imutils
from imutils.video.pivideostream import PiVideoStream

import picamera
import picamera.array

from picamera.array import PiRGBArray
from picamera import PiCamera



sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util
MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'MOV_0004.mp4'
IMAGE_NAME = 'bookshelf.jpg'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
NUM_CLASSES = 1
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
qr_data = ""
centers = []
video = cv2.VideoCapture(PATH_TO_VIDEO)
#cap = cv2.VideoCapture(0)
current = video
Node0, Node1, Node2, Node3 = True, True, True, True
Node1_Frame, Node2_Frame, Node3_Frame = ([0,0],[0,0],[0,0])
#IM_WIDTH, IM_HEIGHT= current.get(cv2.CAP_PROP_FRAME_WIDTH) ,current.get(cv2.CAP_PROP_FRAME_HEIGHT)
IM_WIDTH, IM_HEIGHT= 400,320
lengths = []
RIGHT_L_outside = (int(IM_WIDTH*0.95),int(IM_HEIGHT*0.20))
RIGHT_R_outside = (int(IM_WIDTH*0.6),int(IM_HEIGHT*.80))

CENTER_L_outside = (int(IM_WIDTH*0.58),int(IM_HEIGHT*0.20))
CENTER_R_outside = (int(IM_WIDTH*0.32),int(IM_HEIGHT*.80))

LEFT_L_outside = (int(IM_WIDTH*0.3),int(IM_HEIGHT*0.20))
LEFT_R_outside = (int(IM_WIDTH*0.05),int(IM_HEIGHT*.80))

CURRENT_SPACE = []
MAX_SPACE = []
Node_Frame_Wait_Time = 0
N1, N2, N3 = (0,0,0)
QR_SCAN, FIND_MATCH_SHELF, RETURN_BOOK, GO_BACK = False, False, False, False
font = cv2.FONT_HERSHEY_SIMPLEX

min_t_hold = 0.80

detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def connectArduino():
    arduino = serial.Serial('COM4', 9600, timeout=1)
    time.sleep(2)
    print('Connection to Arduino')
    return arduino
# arduino = connectArduino()


# def led_test():
#     arduino.write(("R{0}C{1}L{2}".format(N3,N2,N1)).encode())

def return_book():
    Node1, Node2, Node3 = True, True, True
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    def node_status():
        if Node1:
            cv2.putText(frame, "NOT AVAILABLE", (LEFT_R_outside),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if Node2:
            cv2.putText(frame, "NOT AVAILABLE", (CENTER_R_outside),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if Node3:
            cv2.putText(frame, "NOT AVAILABLE", (RIGHT_R_outside),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    global  frame
    
    vs = PiVideoStream().start()
    time.sleep(2.0)
    with detection_graph.as_default():
        with tf.Session() as sess:
            while True:
                
                frame = vs.read()
                frame = imutils.resize(frame, width=400)
                t1 = cv2.getTickCount()
                
                cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (30, 50), font, 1, (255, 255, 0), 2,
                            cv2.LINE_AA)
                t2 = cv2.getTickCount()
                time1 = (t2 - t1) / freq
                frame_rate_calc = 1 / time1
              
                frame_expanded = np.expand_dims(frame, axis=0)

                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: frame_expanded})
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=5,
                    min_score_thresh=min_t_hold
                    #min_score_thresh=0.38
                )
                cv2.rectangle(frame, RIGHT_L_outside, RIGHT_R_outside, (255, 255, 255), 1)
                cv2.rectangle(frame, CENTER_L_outside, CENTER_R_outside, (255, 255, 255), 1)
                cv2.rectangle(frame, LEFT_L_outside, LEFT_R_outside, (255, 255, 255), 1)

                try:
                    for i, b in enumerate(boxes[0]):
                        if classes[0][i] == 1:  # if book
                            # if scores[0][i] > 0.5:
                            if scores[0][i] > min_t_hold:
                                ymin = boxes[0][i][0]
                                xmin = boxes[0][i][1]
                                ymax = boxes[0][i][2]
                                xmax = boxes[0][i][3]

                                width = int(((xmin + xmax / 2) * IM_WIDTH))
                                height = int(((ymin + ymax / 2) * IM_HEIGHT))
                                mid_x = (xmax + xmin) / 2  # in percentage
                                mid_y = (ymax + ymin) / 2  # in percentage
                                mid_x_pixel = int(mid_x * IM_WIDTH)
                                mid_y_pixel = int(mid_y * IM_HEIGHT)
                                apx_distance = round((1 - (xmax - xmin)) ** 4, 1)
                                centers.append((mid_x_pixel, mid_y_pixel))
                                # cv2.putText(frame, '{}'.format(apx_distance), (mid_x_pixel,mid_y_pixel), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                                cv2.circle(frame, (mid_x_pixel, mid_y_pixel), 2, (255, 0, 0))
                                # if apx_distance <= 0.5:
                                if apx_distance <= 0.9:  # change to 0.5 ^
                                    cv2.putText(frame, "CLOSE", (mid_x_pixel - 50, mid_y_pixel - 50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (244, 66, 137), 2)
                                    if mid_x > 0.6 and mid_x < 0.95:
                                        cv2.putText(frame, "RIGHT", (mid_x_pixel - 50, mid_y_pixel),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (66, 244, 128), 2)
                                        Node3_Frame[1] += 1  # 1 is True 0 is False
                                        if Node3_Frame[1] >= Node_Frame_Wait_Time:
                                            Node3 = True
                                            N3 = 1
                                            Node3_Frame[1] = 0
                                    elif (mid_x > 0.6 and mid_x < 0.95) == False:
                                        Node3_Frame[0] += 1  # 1 is True 0 is False
                                        if Node3_Frame[0] >= Node_Frame_Wait_Time:
                                            Node3 = False
                                            N3 = 0
                                            Node3_Frame[0] = 0
                                    else:
                                        Node3 = None
                                    if mid_x > 0.32 and mid_x < 0.58:
                                        cv2.putText(frame, "CENTER", (mid_x_pixel - 50, mid_y_pixel),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (66, 244, 128), 2)
                                        Node2_Frame[1] += 1  # 1 is True 0 is False
                                        if Node2_Frame[1] >= Node_Frame_Wait_Time:
                                            Node2 = True
                                            N2 = 1
                                            Node2_Frame[1] = 0

                                    elif (mid_x > 0.32 and mid_x < 0.58) == False:
                                        Node2_Frame[0] += 1
                                        if Node2_Frame[0] >= Node_Frame_Wait_Time:
                                            Node2 = False
                                            N2 = 0
                                            Node2_Frame[0] = 0
                                    else:
                                        Node2 = None
                                    if mid_x > 0.05 and mid_x < 0.3:
                                        cv2.putText(frame, "LEFT", (mid_x_pixel - 50, mid_y_pixel),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (66, 244, 128), 2)
                                        Node1_Frame[1] += 1  # 1 is True 0 is False
                                        if Node1_Frame[1] >= Node_Frame_Wait_Time:
                                            Node1 = True
                                            N1 = 1
                                            Node1_Frame[1] = 0
                                    elif (mid_x > 0.05 and mid_x < 0.3) == False:
                                        Node1_Frame[0] += 1
                                        if Node1_Frame[0] >= Node_Frame_Wait_Time:
                                            Node1 = False
                                            N1 = 0
                                            Node1_Frame[0] = 0
                                    else:
                                        Node1 = None
                                    # led_test()
                            node_status()

                            if (len(centers)) >= 2:
                                recentX = 0
                                recentY = 1
                                center_size = len(centers)
                                for i in range(0, center_size - 1):
                                    # cv2.line(frame, (centers[i]), (centers[i+1]), (0, 255, 0), 2) # for connecting books
                                    distance = np.linalg.norm(int(mid_x_pixel + (mid_x_pixel + IM_HEIGHT))
                                                              / 2 - int(mid_y_pixel + (mid_y_pixel + IM_WIDTH)) / 2)
                                    distance_cm = distance / 12

                                    lengths.append((distance, centers[i], centers[i + 1]))  # lenght point A and point B
                                    if len(lengths) >= 2:  # to find the biggest distance and draw a line between
                                        if lengths[i][0] < lengths[i + 1][0] and lengths[i][0] < lengths[i + 1][
                                            0]:  # 321< 400 305 < 359
                                            CURRENT_SPACE.append(lengths[i + 1])
                                        else:
                                            CURRENT_SPACE.append(lengths[i])
                                        recentPointMax = 0

                                        cv2.line(frame, (CURRENT_SPACE[recentPointMax][1]),
                                                 (CURRENT_SPACE[recentPointMax][2]), (255, 255, 255), 2)
                                        cv2.putText(frame, str(CURRENT_SPACE[recentPointMax][0]),
                                                    (mid_x_pixel, mid_y_pixel),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                                        MAX_SPACE.append(CURRENT_SPACE[recentPointMax])
                                        print(" max distance " + str(MAX_SPACE[recentPointMax][0]) + " wot " + str(
                                            MAX_SPACE[recentPointMax][1]) + " wot " + str(MAX_SPACE[recentPointMax][2]))
                                        print(MAX_SPACE[recentPointMax][1][0])
                                        CURRENT_SPACE.clear()
                                        lengths.clear()

                                # centers.clear()
                                # print(lengths)
                                recentX += 1
                                recentY += 1
                                centers.clear()

                    # node_status()
                    # print("break")
                except Exception as e:
                    print(e)
                    traceback.print_exc()

                # cv2.imshow('object detection', cv2.resize(frame, (800, 480)))

                cv2.imshow('object detection', frame)
                # cv2.waitKey(250)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    vs.stop()
                    break
                
                


def qr_display(frame, decodedObjects):
    # Loop over all decoded objects
    for decodedObject in decodedObjects:
        points = decodedObject.polygon
        # If the points do not form a quad, find convex hull
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points
        n = len(hull)
        for j in range(0, n):
            cv2.line(frame, hull[j], hull[(j + 1) % n], (255, 0, 0), 3)

def decode(frame):
    while True:
        data = ""
        decodedObjects = pyzbar.decode(frame)
        for obj in decodedObjects:
            print('Type : ', obj.type)
            print('Data : ', obj.data, '\n')
            data = obj.data
        return decodedObjects, len(decodedObjects), data







vs = PiVideoStream().start()
time.sleep(2.0)

while True:
   
    frame = vs.read()
            
    qr_display(frame, decode(frame)[0])
    if (decode(frame)[1] == 1):
        QR_SCAN = True
        qr_data = decode(frame)[2]

    cv2.imshow('frame', frame)
    cv2.waitKey(10)
    if QR_SCAN == True:
        vs.stop()
        return_book()
        print("YOOOOOOOOOOOOOOOOO {}".format(qr_data))
        break


         

cv2.destroyAllWindows()


