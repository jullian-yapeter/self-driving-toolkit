'''My Road Detector Program'''
from classes.objectDetector import ObjectDetector
from classes.roadDetector import RoadDetector

def main():
    objectDetector = ObjectDetector()
    print "=== STARTED DOWNLOADING MODEL ==="
    objectDetector.downloadModel() # Please comment out after the first time running this
    print "=== FINISHED DOWNLOADING MODEL ==="
    objectDetector.loadModel()
    roadDetector = RoadDetector(objectDetector)
    roadDetector.runRoadDetector()

if __name__ == '__main__':
    main()
