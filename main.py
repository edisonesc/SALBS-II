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
from playsound import playsound
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

centers = []
video = cv2.VideoCapture(PATH_TO_VIDEO)
cap = cv2.VideoCapture(0)
current = cap
Node1, Node2, Node3 = True, True, True
Node1_Frame, Node2_Frame, Node3_Frame = ([0,0],[0,0],[0,0])
IM_WIDTH, IM_HEIGHT= current.get(cv2.CAP_PROP_FRAME_WIDTH) ,current.get(cv2.CAP_PROP_FRAME_HEIGHT)
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
#QR_SCAN, FIND_MATCH_SHELF, RETURN_BOOK, GO_BACK = False, False, False, False
font = cv2.FONT_HERSHEY_SIMPLEX

SCAN_QR_CODE, MATCH_QR_CODE, LINE_FOLLOWER, FIND_MATCH_SHELF, DETECTED_SPACE, RETURN = False, False, False, False, False, False

qr_data_match = None
qr_data_len = []
qr_data_match_len = []
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def connectArduino():
    arduino = serial.Serial('COM5', 9600, timeout=1)
    time.sleep(2)
    print('Connection to Arduino')
    return arduino

#arduino = connectArduino()
# def led_test():
#     arduino.write(("R{0}C{1}L{2}".format(N3,N2,N1)).encode())
forward, stop = [], []
def Phase_3_Find_Space():
    forward, stop = [], []
    print("PHASE 3: FIND SPACE")
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def node_status():
        if Node3: #right
            cv2.putText(frame, "NOT AVAILABLE R", (RIGHT_R_outside),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if Node2: #center
            cv2.putText(frame, "NOT AVAILABLE C", (CENTER_R_outside),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if Node1: #left
            cv2.putText(frame, "NOT AVAILABLE L", (LEFT_R_outside),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



    def comm_serial_check():
        # if Node3 == False and Node2 == False and Node1:
        #     DETECTED_SPACE = True
        #     print("STOP")
        pass




    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    global  frame
    with detection_graph.as_default():
        with tf.Session() as sess:
            while True:
                time.sleep(0.3)
                t1 = cv2.getTickCount()
                ret, frame = current.read()
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
                    min_score_thresh=0.50
                    #min_score_thresh=0.38
                )
                cv2.rectangle(frame, RIGHT_L_outside, RIGHT_R_outside, (255, 255, 255), 1)
                cv2.rectangle(frame, CENTER_L_outside, CENTER_R_outside, (255, 255, 255), 1)
                cv2.rectangle(frame, LEFT_L_outside, LEFT_R_outside, (255, 255, 255), 1)

                try:
                    for i, b in enumerate(boxes[0]):
                        if classes[0][i] == 1:  # if book
                            # if scores[0][i] > 0.5:
                            if scores[0][i] > 0.5:
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
                                if apx_distance <= 0.5:  # change to 0.5 ^
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

                                           #stop

                                    elif (mid_x > 0.6 and mid_x < 0.95) == False:
                                        Node3_Frame[0] += 1  # 1 is True 0 is False
                                        if Node3_Frame[0] >= Node_Frame_Wait_Time:
                                            Node3 = False
                                            N3 = 0
                                            Node3_Frame[0] = 0

                                            #move forward
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
                                            print("forwaaaaaaaaaaaaard")



                                    elif (mid_x > 0.32 and mid_x < 0.58) == False:
                                        Node2_Frame[0] += 1
                                        if Node2_Frame[0] >= Node_Frame_Wait_Time:
                                            Node2 = False
                                            N2 = 0
                                            Node2_Frame[0] = 0
                                            print("stooooooooop")


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
                    comm_serial_check()
                    data = "0"
                    if Node2 == False and Node3 == False and Node1 == False:


                        if len(stop)  >= 4:
                            print("Stop")
                        else:
                            forward.append(1)


                    else :
                        if len(forward) >= 4:
                            print('forward')
                        else:
                            stop.append(1)

                except Exception as e:
                    print(e)


                # cv2.imshow('object detection', cv2.resize(frame, (800, 480)))

                cv2.imshow('object detection', frame)
                cv2.waitKey(500)
                if DETECTED_SPACE == True:
                    cv2.destroyAllWindows()
                    break

    #


#Scan qr code
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

def decode(frame, arr):
    while True:
        data = ""
        decodedObjects = pyzbar.decode(frame)
        for obj in decodedObjects:
            print('Type : ', obj.type)
            print('Data : ', obj.data, '\n')
            data = obj.data
            arr.append(data)
        return decodedObjects,  data

# mh.get_book_width()

def Phase_1_QR_Scan():
    print("PHASE 1: QR SCAN")
    global qr_data
    while True:
        ret, frame = cap.read()
        qr_display(frame, decode(frame, qr_data_len)[0])
        if len(qr_data_len) > 0: #if qr data is scanned
            SCAN_QR_CODE = True #qr state true
            qr_data = str(qr_data_len[0]) #assign to data
            print(qr_data_len[0]) # prints data
            break


        cv2.imshow('frame', frame)
        cv2.waitKey(10)
    
    playsound('QR_SCANNED.mp3')
    playsound('BOOKCOMPARTMENT.mp3')
#matching
def Phase_2_QR_Match():
    print("PHASE 2: QR MATCH")
    while True:
        global MATCH_QR_CODE
        ret, frame = cap.read()
        qr_display(frame, decode(frame, qr_data_match_len)[0])
        if len(qr_data_match_len) > 0:
            qr_data_match = str(qr_data_match_len[len(qr_data_match_len) - 1])
            qr_data_match_split = qr_data_match.replace("'", '').replace('b', "").split(",") # 1, 2, 3, 4, 5, 6
            qr_data_match_split_lower = []
            qr_data_match_split_higher = []


            for num in range(0, len(qr_data_match_split)):
                if num <= int(len(qr_data_match_split) / 2) - 1:
                    qr_data_match_split_lower.append(qr_data_match_split[num])
                else:
                    qr_data_match_split_higher.append(qr_data_match_split[num])

            qr_data_replace = str(qr_data).replace("'", '').replace('b', "")
            print("LOWER SHELF: {0}".format(qr_data_match_split_lower))
            print("HIGHER SHELF: {0}".format(qr_data_match_split_higher))
            for i in range(len(qr_data_match_split)):
                if qr_data_replace in qr_data_match_split_lower[i]:
                    print("LOWER SHELF: {0} is in {1}".format(qr_data_match_split_lower[i], qr_data_replace))  # prints data match
                    MATCH_QR_CODE =True
                    break
                elif qr_data_replace in qr_data_match_split_higher[i]:
                    print("UPPER SHELF: {0} is in {1}".format(qr_data_match_split_higher[i], qr_data_replace))  # prints data match
                    MATCH_QR_CODE = True
                    break
                else:
                    print("{0} no match to {1} line follower".format(qr_data_match_split[i],qr_data_replace))  # prints data match
                    print("forward")
        if MATCH_QR_CODE == True:
            break
        cv2.imshow('frame', frame)
        cv2.waitKey(10)


# 1
# 1-10
def HW_Arm_Neutral_Position():
    print("HW ARM : NEUTRAL POSITION")
    pass
def Phase_1_5_Grab_Book():
    print("PHASE 1.5: GRAB BOOK")
    pass
def Phase_4_Return_Book():
    print("PHASE 4: RETURN BOOK")
    pass

def Phase_5_Return_Origin():
    print("PHASE 5: RETURN ORIGIN BACKWARDS")
    playsound('RETURNED.mp3')
    pass

Phase_1_QR_Scan() #scans the book
# time.sleep(4) # give time
Phase_1_5_Grab_Book() # grabs book
HW_Arm_Neutral_Position() # neutral position
Phase_2_QR_Match() # find and match with (line follower and collision detection)
#slow down when matched()
Phase_3_Find_Space()
Phase_4_Return_Book()
HW_Arm_Neutral_Position()
Phase_5_Return_Origin()




# return_book()



cv2.destroyAllWindows()