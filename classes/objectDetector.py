import os
import sys
import tarfile
import numpy as np
import cv2
import tensorflow as tf
import six.moves.urllib as urllib
from distutils.version import StrictVersion

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# Uses Tensorflow Object Detection API (Faster RCNN model) to detect cars
class ObjectDetector:
    MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28' #Faster RCNN chosen for the balance of speed and accuracy of region proposals
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

    classID = 3 # Cars are ID number 3
    confThresh = 0.7 # Chose 70% Threshold to filter low confidence detections

    detectionGraph = None
    imageTensor = None
    detBoxes = None
    detScores = None
    detClasses = None

    # If not already downloaded, use this function
    def downloadModel(self,):
        opener = urllib.request.URLopener()
        opener.retrieve(self.DOWNLOAD_BASE + self.MODEL_FILE, self.MODEL_FILE)
        tar_file = tarfile.open(self.MODEL_FILE)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

    # Load the graph from the pretrained frozen Faster RCNN graph
    def loadModel(self,):
        self.detectionGraph = tf.Graph()
        with self.detectionGraph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(self.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        # Get relevant tensors
        self.imageTensor = self.detectionGraph.get_tensor_by_name('image_tensor:0')
        self.detBoxes = self.detectionGraph.get_tensor_by_name('detection_boxes:0')
        self.detScores = self.detectionGraph.get_tensor_by_name('detection_scores:0')
        self.detClasses = self.detectionGraph.get_tensor_by_name('detection_classes:0')

    # Run detection and get bounding boxes, confidence scores, and class names
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

    # Filter the detections based on the set score threshold and desired categories (in this case, Car)
    def filterBoxes(self, confThresh, boxes, scores, classes, categories):
        n = len(classes)
        idxs = []
        for i in range(n):
            if classes[i] in categories and scores[i] >= confThresh:
                idxs.append(i)
        boxes = boxes[idxs, ...]
        scores = scores[idxs, ...]
        classes = classes[idxs, ...]
        return boxes, scores, classes

    # Convert the detected bounding box coordinates back to correspond to original image size
    def convImgCoords(self, boxes, height, width):
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width
        return box_coords

    # Draw red boxes
    def drawBoxes(self, img, boxes, thickness=4):
        for i in range(len(boxes)):
            bottom, left, top, right = boxes[i, ...]
            cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),thickness)

    # Draw all detected desired bounding boxes (of cars)
    def drawDetections(self, img, boxes):
        imshape = img.shape
        box_coords = self.convImgCoords(boxes, imshape[0], imshape[1])
        self.drawBoxes(img, box_coords)
        return img

    # Run detection, filtering, and visualization
    def runObjectDetector(self,img):
        boxes, scores, classes = self.detect(img)
        boxes, scores, classes = self.filterBoxes(self.confThresh, boxes, scores, classes,[self.classID])
        output = self.drawDetections(img,boxes)
        return output
