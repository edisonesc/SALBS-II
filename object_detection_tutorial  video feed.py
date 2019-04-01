#!/usr/bin/env python
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[1]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2 #'1.12.0'
cap = cv2.VideoCapture(0)
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

# # Model preparation

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[ ]:



# What model to download.
# MODEL_NAME = 'faster_rcnn_inception_v2_coco'
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_NAME = 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# VIDEO_NAME = 'MOV_0002.mp4'
VIDEO_NAME = 'MOV_619.mkv'
CWD_PATH = os.getcwd()
PATH_TO_VIDEO = os.path.join(CWD_PATH,VIDEO_NAME)
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

# ## Download Model

# In[ ]:


# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)

# tar_file = tarfile.open(MODEL_FILE)
tar_file = tarfile.open('faster_rcnn_inception_v2_coco.tar.gz')

for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

# ## Load a (frozen) Tensorflow model into memory.

# In[ ]:


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[ ]:


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# ## Helper code

# In[ ]:


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[ ]:


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# In[ ]:

centers= []

video = cv2.VideoCapture(PATH_TO_VIDEO)
IM_WIDTH, IM_HEIGHT= video.get(cv2.CAP_PROP_FRAME_WIDTH) ,video.get(cv2.CAP_PROP_FRAME_HEIGHT)
with detection_graph.as_default():
    with tf.Session() as sess:
        # Get handles to input and output tensors
        # while True:
        while video.isOpened():
            # ret, image_np = cap.read()
            ret, image_np = video.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.


            vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    # min_score_thresh=0.01
            )

            try:
                for i, b in enumerate(boxes[0]):
                    if classes[0][i] == 84:  # if book
                        if scores[0][i] > 0.40:
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
                            cv2.circle(image_np, (mid_x_pixel, mid_y_pixel), 2, (255, 0, 0))
                            if apx_distance <= 0.5:
                                cv2.putText(image_np, "CLOSE", (mid_x_pixel - 50, mid_y_pixel - 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
                                if mid_x > 0.4 and mid_x < 0.6:
                                    cv2.putText(image_np, "CENTER", (mid_x_pixel - 50, mid_y_pixel),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

                        if (len(centers)) >= 2:
                            recentX = 0
                            recentY = 1
                            center_size = len(centers)
                            for i in range(0, center_size - 1):
                                cv2.line(image_np, (centers[i]), (centers[i + 1]), (0, 255, 0), 2)
                                distance = np.linalg.norm(int(mid_x_pixel + (mid_x_pixel + IM_HEIGHT))
                                                          / 2 - int(mid_y_pixel + (mid_y_pixel + IM_WIDTH)) / 2)
                                distance_cm = distance / 12
                                cv2.putText(image_np, str(round(distance_cm, 2)) + " cm",
                                            (mid_x_pixel, mid_y_pixel),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

                            centers.clear()
                            recentX += 1
                            recentY += 1
                            centers.clear()

                print("break")
            except Exception as e:
                print(e)

            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break



# In[ ]:


# In[ ]:
