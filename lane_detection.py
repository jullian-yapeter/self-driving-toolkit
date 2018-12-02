'''My Lane Detection Program'''
import os
import math
import numpy as np
import cv2
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from multiprocessing import Process, Manager, Queue, Pool

class LaneDetector():

    args = None

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
        claheimg = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
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
            pass
        return line_img

    # Used to fuse image of extracted lines back with the raw image
    def fuseImages(self, img, initial_img, a, b, l):
        return cv2.addWeighted(initial_img, a, img, b, l)

    # Use pixel value thresholding to isolate white and yellow pixels
    # The assumption is that lane lines are always either white or yellow in colour
    def maskWhiteYellow(self, grayimg, rgbimg, hlsimg):
        # Range of yellow colour in HLS space
        hlsyellowLow = np.array([20, 120, 80], dtype = "uint8")
        hlsyellowHigh = np.array([45, 200, 255], dtype="uint8")
        # Range of yellow colour in HSV space
        rgbyellowLow = np.array([100, 100, 100], dtype = "uint8")
        rgbyellowHigh = np.array([45, 200, 255], dtype= "uint8")
        # Create masks for each colour
        yellowMask1 = cv2.inRange(hlsimg, hlsyellowLow, hlsyellowHigh)
        yellowMask2 = cv2.inRange(rgbimg, rgbyellowLow, rgbyellowHigh)
        whiteMask = cv2.inRange(grayimg, 200, 255) # Range of white in 1 channel
        yellowMask = cv2.bitwise_or(yellowMask1, yellowMask2)
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
        #claheimg = self.claheEqualization(img)
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hlsimg = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2HLS)
        maskedImg = self.maskWhiteYellow(grayimg, rgbimg, hlsimg)
        gaussimg = cv2.GaussianBlur(maskedImg,(5,5),0)
        cannyimg = self.cannyDetector(gaussimg)
        roi_image = self.regionOfInterest(cannyimg)
        outimg = self.omitSmallContours(roi_image)
        line_image = self.houghLines(outimg)
        #result = self.fuseImages(line_image, cv2.resize(detectimg,(0,0),fx=2,fy=2,interpolation=cv2.INTER_CUBIC), a=0.8, b=1.0, l=0.0)
        result = self.fuseImages(line_image, img, a=0.8, b=1.0, l=0.0)
        return result

    # Run as a multiprocess program
    # Images are read from the video in a separate process
    def runLaneDetector(self,):
        proc_manager = Manager()
        raw_q = proc_manager.Queue() # Raw images are pushed to a queue waiting to be processed
        videocap_proc = Process(target = self.videocap, args = (raw_q,))
        videocap_proc.start()
        starttime = time.time()
        while raw_q.qsize() <= 0: # Wait until the first image is pushed to the queue
            pass
        if raw_q.qsize() > 0:
            img = raw_q.get()
            img = cv2.resize(img,(1280, 740))
            imshape = img.shape # Need shape of image to initialize opencv video writer
            writer = cv2.VideoWriter('lane_detection_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 15, (imshape[1],imshape[0]), True)
        while True:
            if raw_q.qsize() > 0:
                starttime = time.time()
                img = raw_q.get()
                img = cv2.resize(img,(1280, 740))
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

def main():
    laneDetector = LaneDetector()
    laneDetector.runLaneDetector()


if __name__ == "__main__":
    main()
