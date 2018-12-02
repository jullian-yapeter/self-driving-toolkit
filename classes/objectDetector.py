import os
import sys
import math
import time
import argparse
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from multiprocessing import Process, Manager, Queue, Pool
import cv2
import tensorflow as tf
from PIL import ImageColor
from PIL import Image
from PIL import ImageDraw
import matplotlib.image as mpimg
import six.moves.urllib as urllib
from distutils.version import StrictVersion
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class ObjectDetector:
    MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

    classID = 3 # Cars are ID number 3
    confThresh = 0.7

    detectionGraph = None
    imageTensor = None
    detBoxes = None
    detScores = None
    detClasses = None

    def downloadModel(self,): #If not already downloaded
        opener = urllib.request.URLopener()
        opener.retrieve(self.DOWNLOAD_BASE + self.MODEL_FILE, self.MODEL_FILE)
        tar_file = tarfile.open(self.MODEL_FILE)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

    def loadModel(self,):
        self.detectionGraph = tf.Graph()
        with self.detectionGraph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(self.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        self.imageTensor = self.detectionGraph.get_tensor_by_name('image_tensor:0')
        self.detBoxes = self.detectionGraph.get_tensor_by_name('detection_boxes:0')
        self.detScores = self.detectionGraph.get_tensor_by_name('detection_scores:0')
        self.detClasses = self.detectionGraph.get_tensor_by_name('detection_classes:0')

    def detect(self,img,runs=1):
        img = np.expand_dims(np.asarray(img, dtype=np.uint8), 0)
        with tf.Session(graph=self.detectionGraph) as sess:
            for i in range(runs):
                (boxes, scores, classes) = sess.run([self.detBoxes, self.detScores, self.detClasses],
                                                    feed_dict={self.imageTensor: img})
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            return boxes, scores, classes

    def filter_boxes(self, min_score, boxes, scores, classes, categories):
        n = len(classes)
        idxs = []
        for i in range(n):
            if classes[i] in categories and scores[i] >= min_score:
                idxs.append(i)
        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        return box_coords

    def draw_boxes(self, img, boxes, classes, thickness=4):
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])
            cv2.rectangle(img,(left,top),(right,bot),(0,0,255),thickness)

    def to_image_coords(self, boxes, height, width):
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        return box_coords

    def illustrate_detection(self, image, boxes, classes):
        imshape = image.shape
        box_coords = self.to_image_coords(boxes, imshape[0], imshape[1])
        self.draw_boxes(image, box_coords, classes)
        return image

    def runObjectDetector(self,img):
        boxes, scores, classes = self.detect(img)
        boxes, scores, classes = self.filter_boxes(self.confThresh, boxes, scores, classes,[self.classID])
        out = self.illustrate_detection(img,boxes,classes)
        return out
