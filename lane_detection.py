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

    def claheEqualization(self, img):
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        claheimg = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return claheimg

    def regionOfInterest(self, img):
        imshape = img.shape
        lower_left = (imshape[1]/9,imshape[0])
        lower_right = (imshape[1]-imshape[1]/9,imshape[0])
        top_left = (imshape[1]/2-imshape[1]/8,imshape[0]/2+imshape[0]/10)
        top_right = (imshape[1]/2+imshape[1]/8,imshape[0]/2+imshape[0]/10)
        vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
        #defining a blank mask to start with
        mask = np.zeros(img.shape[:2], dtype="uint8")

        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        #filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def drawLines(self, img, lines, color=[255, 0, 0], thickness=2):
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def houghLines(self, img):
        rho = 2
        theta = np.pi/180
        #threshold is minimum number of intersections in a grid for candidate line to go to output
        threshold = 50
        min_line_len = 70
        max_line_gap = 200
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        try:
            self.drawLines(line_img, lines)
        except Exception as e:
            pass
        return line_img

    def fuseImages(self, img, initial_img, a, b, l):
        return cv2.addWeighted(initial_img, a, img, b, l)

    def maskWhiteYellow(self,grayimg, hsvimg):
        lower_yellow = np.array([20, 100, 100], dtype = "uint8")
        upper_yellow = np.array([30, 255, 255], dtype="uint8")

        mask_yellow = cv2.inRange(hsvimg, lower_yellow, upper_yellow)
        mask_white = cv2.inRange(grayimg, 200, 255)
        mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
        mask_yw_image = cv2.bitwise_and(grayimg, mask_yw)
        return mask_yw_image

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

    def cannyDetector(self, img):
        low_threshold = 50
        high_threshold = 150
        cannyimg = cv2.Canny(img,low_threshold,high_threshold)
        return cannyimg

    def processImg(self,img):
        claheimg = self.claheEqualization(img)
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsvimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask_yw_image = self.maskWhiteYellow(grayimg, hsvimg)
        gaussimg = cv2.GaussianBlur(mask_yw_image,(5,5),0)
        cannyimg = self.cannyDetector(gaussimg)
        roi_image = self.regionOfInterest(cannyimg)
        outimg = self.omitSmallContours(roi_image)
        line_image = self.houghLines(outimg)
        result = self.fuseImages(line_image, img, a=0.8, b=1.0, l=0.0)
        return result

    def runLaneDetector(self,):
        #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #out = cv2.VideoWriter('lane_detection_output.avi',fourcc, 20.0, (740,1280))
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
            writer = cv2.VideoWriter('lane_detection_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (imshape[1],imshape[0]), True)
        while True:
            if raw_q.qsize() > 0:
                starttime = time.time()
                img = raw_q.get()
                proc_img = self.processImg(img)
                cv2.imshow('result',proc_img)
                writer.write(proc_img)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
            elif time.time() - starttime >= 1:
                break
        writer.release()
        cv2.destroyAllWindows()
        videocap_proc.terminate()
        videocap_proc.join()

def main():
    laneDetector = LaneDetector()
    laneDetector.runLaneDetector()


if __name__ == "__main__":
    main()
