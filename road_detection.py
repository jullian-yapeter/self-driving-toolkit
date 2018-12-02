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
from PIL import ImageColor
from PIL import Image
from PIL import ImageDraw
import matplotlib.image as mpimg
import six.moves.urllib as urllib
from distutils.version import StrictVersion
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from classes.objectDetector import ObjectDetector
from classes.roadDetector import RoadDetector

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

def main():
    objectDetector = ObjectDetector()
    objectDetector.loadModel()
    #objectDetector.downloadModel()
    roadDetector = RoadDetector(objectDetector)
    roadDetector.runRoadDetector()

if __name__ == '__main__':
    main()
