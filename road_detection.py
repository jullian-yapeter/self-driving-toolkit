'''My Road Detector Program'''
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
import matplotlib.image as mpimg
import six.moves.urllib as urllib
from distutils.version import StrictVersion
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

class RoadDetector:

    args = None
    objectDetector = None # Uses object of ObjectDetector class to detect cars

    def __init__(self,od):
        self.objectDetector = od

    # Retrieve user inputted video file path
    def argparser(self,):
        ap = argparse.ArgumentParser()
        ap.add_argument("--vidfile", required = True, help = "Enter your test video file")
        self.args = vars(ap.parse_args())

    def videocap(self,raw_q):
        self.argparser()
        cap = cv2.VideoCapture(self.args["vidfile"])
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                raw_q.put(frame)
            else:
                break
        cap.release()

    # CLAHE histogram equalization is used to amplify contrast while minimizing the amplification of noise
    def claheEqualization(self, img):
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        claheimg = cv2.cvtColor(limg, cv2.COLOR_BGR2LAB)
        return claheimg

    # Create a trapezoidal region of interest, and mask out everything outside the ROI
    # Thus minimizing the possibility of "lanes" being detected in implausible areas
    def regionOfInterest(self, img):
        imshape = img.shape
        lower_left = (imshape[1]/10,imshape[0])
        lower_right = (imshape[1]-imshape[1]/10,imshape[0])
        top_left = (imshape[1]/2-imshape[1]/7,imshape[0]/2+imshape[0]/10)
        top_right = (imshape[1]/2+imshape[1]/7,imshape[0]/2+imshape[0]/10)
        vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
        #defining a blank mask to start with
        mask = np.zeros(img.shape[:2], dtype="uint8")

        # Fill 1 or 3 channels of masking
        if len(img.shape) > 2:
            channels = img.shape[2]  # i.e. 3 or 4 depending on your image
            maskcolour = (255,) * channels
        else:
            maskcolour = 255
        cv2.fillPoly(mask, vertices, maskcolour)
        maskedRoi = cv2.bitwise_and(img, mask)

        return maskedRoi

    # Draw the detected lane lines or visibility
    def drawLines(self, img, lines, colour=[0, 255, 0], thickness=3): # Because Green is my favourite colour
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), colour, thickness)

    # Use Hough Transform algorithm to use the detected lines to extrapolate lines (i.e. voting system)
    def houghLines(self, img):
        rho = 2
        theta = np.pi/180
        minNumOfIntersections = 50
        minLen = 70
        maxGap = 200
        lines = cv2.HoughLinesP(img, rho, theta, minNumOfIntersections, np.array([]), minLineLength=minLen, maxLineGap=maxGap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        try:
            self.drawLines(line_img, lines)
        except Exception as e:
            print "Error drawing lines: ", e
            pass
        return line_img

    # Used to fuse image of extracted lines back with the raw image
    def fuseImages(self, img, initial_img, a, b, l):
        return cv2.addWeighted(initial_img, a, img, b, l)

    # Use pixel value thresholding to isolate white and yellow pixels
    # The assumption is that lane lines are always either white or yellow in colour
    def maskWhiteYellow(self, grayimg, hsvimg):
        # Range of yellow colour in HSV space
        yellowLow = np.array([20, 100, 100], dtype = "uint8")
        yellowHigh = np.array([30, 255, 255], dtype="uint8")
        # Create masks for each colour
        yellowMask = cv2.inRange(hsvimg, yellowLow, yellowHigh)
        whiteMask = cv2.inRange(grayimg, 200, 255) # Range of white in 1 channel
        # Fuse the two masks
        wyMask = cv2.bitwise_or(whiteMask, yellowMask)
        # Apply the mask
        maskedImg = cv2.bitwise_and(grayimg, wyMask)

        return maskedImg

    # This function is used to omit noise;
    # Remove thresholded pixels that are too small to constitute lane lines
    def omitSmallContours(self, img):
        mask = np.ones(img.shape[:2], dtype="uint8") * 255
        # find all contours, loop through them, and mask the contours that have too small of an area
        _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if area < 20 or w < 4 or h < 4:
                cv2.drawContours(mask, [c], -1, 0, -1)
        # bitwise AND with mask to omit unwanted areas
        out = cv2.bitwise_and(img, mask)
        return out

    # Use Canny edge detection (gradient based detection)
    def cannyDetector(self, img):
        low_threshold = 50
        high_threshold = 150
        cannyimg = cv2.Canny(img,low_threshold,high_threshold)
        return cannyimg

    # Process the raw images in the correct order using the above processing functions
    def processImg(self,img):
        #detectimg = self.objectDetector.outputDetections(cv2.resize(img,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC))
        detectimg = self.objectDetector.outputDetections(img)
        laheimg = self.claheEqualization(img)
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsvimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        maskedImg = self.maskWhiteYellow(grayimg, hsvimg)
        gaussimg = cv2.GaussianBlur(maskedImg,(5,5),0)
        cannyimg = self.cannyDetector(gaussimg)
        roi_image = self.regionOfInterest(cannyimg)
        outimg = self.omitSmallContours(roi_image)
        line_image = self.houghLines(outimg)
        #result = self.fuseImages(line_image, cv2.resize(detectimg,(0,0),fx=2,fy=2,interpolation=cv2.INTER_CUBIC), a=0.8, b=1.0, l=0.0)
        result = self.fuseImages(line_image, detectimg, a=0.8, b=1.0, l=0.0)
        return result

    # Run as a multiprocess program
    # Images are read from the video in a separate process
    def runRoadDetector(self,):
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter('road_detection_output.avi',fourcc, 20.0, (740,1280))
        proc_manager = Manager()
        raw_q = proc_manager.Queue()
        videocap_proc = Process(target = self.videocap, args = (raw_q,))
        videocap_proc.start()
        starttime = time.time()
        while raw_q.qsize() <= 0:
            pass
        if raw_q.qsize() > 0:
            img = raw_q.get()
            imshape = img.shape
            writer = cv2.VideoWriter('road_detection_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (imshape[1],imshape[0]), True)
        while True:
            if raw_q.qsize() > 0:
                starttime = time.time()
                img = raw_q.get()
                proc_img = self.processImg(img)
                cv2.imshow('result',proc_img)
                writer.write(proc_img)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            elif time.time() - starttime >= 1: # Stop the program 1 second after the video ends
                break
        cv2.destroyAllWindows()
        writer.release()
        videocap_proc.terminate()
        videocap_proc.join()

class ObjectDetector:
    MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    category_index = None
    detection_graph = None

    def downloadModel(self,): #If not already downloaded
        opener = urllib.request.URLopener()
        opener.retrieve(self.DOWNLOAD_BASE + self.MODEL_FILE, self.MODEL_FILE)
        tar_file = tarfile.open(self.MODEL_FILE)
        for file in tar_file.getmembers():
          file_name = os.path.basename(file.name)
          if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

    def loadModel(self,):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(self.PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    def setCategoryIndex(self,):
        self.category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS, use_display_name=True)

    def runInference(self, image, graph):
      with graph.as_default():
        with tf.Session() as sess:
          ops = tf.get_default_graph().get_operations()
          all_tensor_names = {output.name for op in ops for output in op.outputs}
          tensor_dict = {}
          for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
          ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
              tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                  tensor_name)
          if 'detection_masks' in tensor_dict:

            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)

            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
          image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

          output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})

          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
              'detection_classes'][0].astype(np.uint8)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
      return output_dict

    def outputDetections(self,image):
        output_dict = self.runInference(image, self.detection_graph)
        vis_util.visualize_boxes_and_labels_on_image_array(
          image,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          self.category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=8)
        return image

def main():
    objectDetector = ObjectDetector()
    #objectDetector.downloadModel()
    objectDetector.loadModel()
    objectDetector.setCategoryIndex()
    roadDetector = RoadDetector(objectDetector)
    roadDetector.runRoadDetector()

if __name__ == '__main__':
    main()
