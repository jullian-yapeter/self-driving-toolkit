I'm Jullian, a third year Mechatronics student from the University of Waterloo
with a passion for robotics, computer vision, and AI

Solution:

I chose to complete the self-driving coding challenge. For the challenge, I
created a lane + car detection program.

For lane detection, I used pure computer vision. Using colour range thresholding
in multiple colour spaces, edge detection, Hough transform, ROI isolation, and
contour finding techniques, I was able to detect white and yellow lane lines
from a test video I found online.

For car detection, I opted to use Tensorflow's object detection API with a
Faster RCNN model pre-trained on the COCO dataset. I used Faster RCNN for its
balanced attributes of speed and accuracy. Detections are filtered based
on a confidence threshold and detected class name (only show detected cars). I
did not include the Faster RCNN model in this repository, but I did include a
download function that allows you to pull the model from Tensorflow's online
source (Simply comment out the download function on line 8 of road_detection.py
after the first time running it).

I was using the CPU version of Tensorflow due to my lack of resources. Therefore,
the detections are relatively slow. However, I included a video of what it would
look like if it was run on a GPU.

Videos of the program working:

In the case that the program does not work on your end due to unforeseen reasons.
Please refer to the videos_of_the_program_working folder to watch videos of when
it does work.

How to run the program:

I included two scripts in root: road_detection.py and lane_detection.py
as well as two classes in /classes: objectDetector.py and roadDetector.py

Unzip the provided example.mp4 video

road_detection.py uses the two classes to perform both lane detection and car
detection. Pass it a video file as an argument to run it.

In the project folder, self-driving-toolkit, run:

python road_detection.py --vidfile example.mp4

A window will pop up and begin to play the processed video showing green lines
to indicate the detected lanes and red boxes to indicate detected cars.

I also included a script to run just the lane detection which uses less
dependencies (In case road_detection.py fails to run)

lane_detection.py uses performs lane detection only.
Pass it a video file as an argument to run it.

In the project folder, self-driving-toolkit, run:

python lane_detection.py --vidfile example.mp4

A window will pop up and begin to play the processed video showing green lines
to indicate the detected lanes.

Dependencies:

- Python 2.7
- numpy
- OpenCV
- Tensorflow >= 1.9.0
- A test video file

Tensorflow Object Detection API dependencies:
- Protobuf 3.0.0
- Python-tk
- Pillow 1.0
- lxml
- tf Slim (which is included in the "tensorflow/models/research/" checkout)
- Jupyter notebook
- Matplotlib
- Cython
- contextlib2
- cocoapi

If the Tensorflow object detection API is not already installed,
please follow these instructions:
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

Please ask me any questions if the program does not run properly
