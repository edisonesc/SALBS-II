import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from PIL import Image
import serial
import time
import pyzbar.pyzbar as pyzbar

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
cap = cv2.VideoCapture(0)


current = cap
Node1, Node2, Node3 = True, True, True
Node1_Frame, Node2_Frame, Node3_Frame = ([0,0],[0,0],[0,0])
IM_WIDTH, IM_HEIGHT= current.get(cv2.CAP_PROP_FRAME_WIDTH) ,current.get(cv2.CAP_PROP_FRAME_HEIGHT)
lengths = []



detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def get_book_width():
    with detection_graph.as_default():
        with tf.Session() as sess:
            while True:
                ret, frame = current.read()

                frame_expanded = np.expand_dims(frame, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
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
                    # min_score_thresh=0.80
                    min_score_thresh=0.90
                )
                for i, b in enumerate(boxes[0]):
                    if classes[0][i] == 1:  # if book
                        # if scores[0][i] > 0.5:
                        if scores[0][i] > 0.4:
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
                            if apx_distance <= 0.9:
                                print(apx_distance)
                                print(width)


                cv2.imshow('object detection', frame)
                # cv2.waitKey(250)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()

                    break


def line_follower():
    while (True):
        # Capture the frames
        ret, frame = current.read()
        # Crop the image
        crop_img = frame[90:5, 10:160]
        # Convert to grayscale

        # ret, frame = frame.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Color thresholding
        ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
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
                # led_test()

                # print("CX: {0} || CY: {1}".format(cx, cy))
            except Exception as e:
                print(e)

            # cv2.line(frame, (640, 0), (640, 1080), (255, 0, 0), 1)
            # cv2.line(frame, (1280, 0), (1280, 1080), (255, 0, 0), 1)

        else:

            print("I don't see the line")

        # Display the resulting frame

        # cv2.imshow('frame', crop_img)
        cv2.imshow('object detection', cv2.resize(frame, (800, 480)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break