# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

from sort import Sort

tracker = Sort()
memory = {}
#line1 = [(300, 350), (1350, 350)]
#line2 = [(300, 525), (1350, 525)]
#line3 = [(300, 700), (1350, 700)]

line1 = [(45,68), (300, 68)]
line2 = [(45, 134), (300, 134)]
line3 = [(45, 184), (300, 184)]
entered = 0
left = 0
middle=0


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debugger",default=True, help="provide TRUE to debug the code")
# ap.add_argument("-o", "--output", required=True,
#    help="path to output video")
# ap.add_argument("-y", "--yolo", required=True,
#    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())



VIDEO_NAME = 'input/Test6_RAW-01.avi'
#
MODEL_NAME = 'yolo-coco'
OUT_VIDEO_NAME = 'testop.avi'
## Grab path to current working directory
CWD_PATH = os.getcwd()
#
PATH_TO_CKPT1 = os.path.join(CWD_PATH, MODEL_NAME, 'coco.names')
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME)
## Path to video
PATH_TO_VIDEO = os.path.join(CWD_PATH, VIDEO_NAME)
#
PATH_TO_OUT_VIDEO = os.path.join(CWD_PATH, OUT_VIDEO_NAME)


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

trackerDict = {}


#def oneThenTwo(p0, p1, id):
#    id = str(id)
#    if args["debugger"]:
#        print("oneThenTwo: " + str(trackerDict))
#    if id+"_oneThenTwo" not in list(trackerDict.keys()):
#        trackerDict[id+"_oneThenTwo"] = "Did not enter the store"
#    if intersect(p0, p1, line1[0], line1[1]):
#        trackerDict[id+"_oneThenTwo"] =  "Crossed line 1"
#    if id+"_oneThenTwo" in list(trackerDict.keys()) and trackerDict[id + "_oneThenTwo"] !=  "Did not enter the store":
#        if intersect(p0, p1, line2[0], line2[1]):
#            trackerDict[id + "_oneThenTwo"] = "Crossed line 2"
#    return trackerDict[id+"_oneThenTwo"]
#
#
#def twoThenOne(p1, p0, id):
#    id = str(id)
#    if args["debugger"]:
#        print("twoThenOne: " + str(trackerDict))
#    if id + "_twoThenOne" not in list(trackerDict.keys()):
#        trackerDict[id + "_twoThenOne"] = "Did not leave the store"
#    if intersect(p0, p1, line2[0], line2[1]):
#        trackerDict[id + "_twoThenOne"] = "Crossed line 2"
#    if id + "_twoThenOne" in list(trackerDict.keys()) and trackerDict[id + "_twoThenOne"] !=  "Did not leave the store" :
#        if intersect(p0, p1, line1[0], line1[1]):
#            trackerDict[id + "_twoThenOne"] = "Crossed line 1"
#    return trackerDict[id + "_twoThenOne"]



def oneThenTwo(p0, p1, id):
    id = str(id)
    if id+"_oneThenTwo" not in trackerDict.keys():
        trackerDict[id+"_oneThenTwo"] = "test"
        #trackerDict[id+"_twoThenOne"] = "test"
        #trackerDict[id+"_threeThenOne"] = "test"
    if trackerDict[id+"_oneThenTwo"] != "Objcrossed":   
      if args["debugger"]:
         print("oneThenTwo: " + str(trackerDict))
      #if id+"_oneThenTwo" not in list(trackerDict.keys()):
       #   trackerDict[id+"_oneThenTwo"] = "Did not enter the store"
      if intersect(p0, p1, line1[0], line1[1]):
          trackerDict[id+"_oneThenTwo"] =  "Crossed line 1"
      if  trackerDict[id+"_oneThenTwo"] == "Crossed line 1":    
      #if trackerDict[id+"_oneThenTwo"] == "Crossed line 1" :
          if intersect(p0, p1, line2[0], line2[1]):
              trackerDict[id + "_oneThenTwo"] = "Crossed line 2" 
      elif id+"_twoThenOne" in list(trackerDict.keys()) and trackerDict[id+"_twoThenOne"] == "Crossed line 2" :
             if intersect(p0, p1, line3[0], line3[1]):
                trackerDict[id + "_oneThenTwo"] = "Crossed line 3"    
      if  trackerDict[id + "_oneThenTwo"] == "Crossed line 2" or trackerDict[id+"_oneThenTwo"] == "Crossed line 3": 
              trackerDict[id + "_oneThenTwo"] = "Objcrossed"          

      return trackerDict[id+"_oneThenTwo"]


def twoThenOne(p1, p0, id):
    id = str(id)
    print("trackerDict.keys()",trackerDict.keys())
    if id+"_twoThenOne" not in trackerDict.keys():
        trackerDict[id+"_twoThenOne"] = "test"
    if trackerDict[id+"_twoThenOne"] != "Objcrossed":
       if args["debugger"]:
           print("twoThenOne: " + str(trackerDict))

       if intersect(p0, p1, line2[0], line2[1]):
           trackerDict[id + "_twoThenOne"] = "Crossed line 2"
#       if id + "_twoThenOne" in list(trackerDict.keys()) and trackerDict[id + "_twoThenOne"] !=  "test" :
#           if intersect(p0, p1, line1[0], line1[1]):
#               trackerDict[id + "_twoThenOne"] = "Crossed line 1"
       return trackerDict[id + "_twoThenOne"]

def threeThenOne(p1, p0, id):
    id = str(id)
    print("trackerDict.keys()",trackerDict.keys())
    if id+"_threeThenOne" not in trackerDict.keys():
        trackerDict[id+"_threeThenOne"] = "test"
    if trackerDict[id+"_threeThenOne"] != "Objcrossed":
       if args["debugger"]:
           print("threeThenOne: " + str(trackerDict))
#       if id + "_twoThenOne" not in list(trackerDict.keys()):
#           trackerDict[id + "_twoThenOne"] = "Did not leave the store"
       if intersect(p0, p1, line3[0], line3[1]):
           trackerDict[id + "_threeThenOne"] = "Crossed line 3"
       if id+"_twoThenOne" in list(trackerDict.keys()) and trackerDict[id+"_threeThenOne"] == "Crossed line 3" :
          if intersect(p0, p1, line2[0], line2[1]):
              trackerDict[id + "_threeThenOne"] = "Crossed line 2"
       elif trackerDict[id+"_twoThenOne"] == "Crossed line 2" :
             if intersect(p0, p1, line1[0], line1[1]):
                trackerDict[id + "_threeThenOne"] = "Crossed line 1"
       if  trackerDict[id + "_threeThenOne"] == "Crossed line 2" or trackerDict[id+"_threeThenOne"] == "Crossed line 1": 
              trackerDict[id + "_threeThenOne"] = "Objcrossed" 
       return trackerDict[id + "_threeThenOne"]
   
 
    


# load the COCO class labels our YOLO model was trained on
# labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
# LABELS = open(labelsPath).read().strip().split("\n")

labelsPath = os.path.sep.join([PATH_TO_CKPT, "coco.names"])
# print(labelsPath)
LABELS = open(PATH_TO_CKPT1).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
# weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
# configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
weightsPath = os.path.sep.join([PATH_TO_CKPT, "frozen_inference_graph.pb"])
configPath = os.path.sep.join([PATH_TO_CKPT, "mscoco_label_map.pbtxt"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
#print("OpenCV Version: {}".format(cv2.__version__))
#net = cv2.dnn.readNetFromDarknet(configPath, weightsPath,)
net = cv2.dnn.readNetFromTensorflow(weightsPath,configPath)
ln = net.getLayerNames()
print("ln:",ln)
#ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
# vs = cv2.VideoCapture(args["input"])
vs = cv2.VideoCapture(PATH_TO_VIDEO)
writer = None
(W, H) = (None, None)

frameIndex = 0

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# loop over frames from the video file stream
while True:

    a = None
    b = None
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # cv2.imshow('frame1',frame)
    # cv2.waitKey(1000)

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    #adjusted = adjust_gamma(frame, gamma=1.5)
    
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward()
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:

        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            #if classID==0:
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # print(counter)

                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                     box = detection[0:4] * np.array([W, H, W, H])
                     (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                     x = int(centerX - (width / 2))
                     y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                     boxes.append([x, y, int(width), int(height)])
                     confidences.append(float(confidence))
                     classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    dets = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            dets.append([x, y, x + w, y + h, confidences[i]])

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)

    boxes = []
    indexIDs = []
    c = []
    previous = memory.copy()
    memory = {}

    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        indexIDs.append(int(track[4]))
        memory[indexIDs[-1]] = boxes[-1]

    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            if args["debugger"]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            # cv2.rectangle(frame, (x, y), (w, h), color, 2)

            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))
                p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                # print("p0,p1", p0, p1)
                if args["debugger"]:
                    cv2.line(frame, p0, p1, (0,0,0), 3)
                    # cv2.line(frame, (80, H // 2), (404, H // 2), color, 3)
                    print('*******************************************************')
                    #if LABELS[classIDs[i]] == "person":
                    print("Person Detected")
                    
                    a = oneThenTwo(p0, p1, indexIDs[i])
                    b = twoThenOne(p0, p1, indexIDs[i])
                    c = threeThenOne(p0, p1, indexIDs[i])

                if a == "Objcrossed":
                    entered = entered + 1
                    #del trackerDict[str(indexIDs[i]) + "_oneThenTwo"]
                    #del trackerDict[str(indexIDs[i]) + "_twoThenOne"]
                    trackerDict[str(indexIDs[i]) + "_oneThenTwo"]="Objcrossed"
                    trackerDict[str(indexIDs[i]) + "_twoThenOne"]="Objcrossed"
                    trackerDict[str(indexIDs[i]) + "_threeThenOne"]="Objcrossed"  

                if c == "Objcrossed":
                    left = left + 1
                    trackerDict[str(indexIDs[i]) + "_oneThenTwo"]="Objcrossed"
                    trackerDict[str(indexIDs[i]) + "_twoThenOne"]="Objcrossed"  
                    trackerDict[str(indexIDs[i]) + "_threeThenOne"]="Objcrossed"  
                   

            text = "{}:{}: {:.4f}".format(indexIDs[i], LABELS[classIDs[i]], confidences[i])
            # text = "{}".format(indexIDs[i])
            if args["debugger"]:
                cv2.putText(frame, text+" oneThenTwo: "+str(a)+" twoThenOne: "+str(b)+" "+str(trackerDict), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (191, 39, 19), 2)

            i += 1

    # draw line
    if args["debugger"]:
        print("dummy")
        cv2.line(frame, line1[0], line1[1], (255, 255, 255), 5)
        cv2.line(frame, line2[0], line2[1], (0, 0, 0), 5)
        cv2.line(frame, line3[0], line3[1], (255, 255, 255), 5)

    # draw counter
    cv2.putText(frame, "Entered: "+ str(entered), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (25, 92, 6), 3)
    #cv2.putText(frame, "Entered: "+ str(middle), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (92, 33, 6), 3)
    cv2.putText(frame, "Left: "+ str(left), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (92, 33, 6), 3)
    #cv2.putText(frame, "Entered: "+ str(entered), (100, 200), cv2.FONT_HERSHEY_DUPLEX, 3.0, (25, 92, 6), 3)
    #cv2.putText(frame, "Left: "+ str(left), (100, 400), cv2.FONT_HERSHEY_DUPLEX, 3.0, (92, 33, 6), 3)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        # writer = cv2.VideoWriter(args["output"], fourcc, 30,
        #   (frame.shape[1], frame.shape[0]), True)
        writer = cv2.VideoWriter(PATH_TO_OUT_VIDEO, fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)
      # show the output frame
    if args["debugger"]:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF


    # write the output frame to disk
    writer.write(frame)

    # increase frame index
    frameIndex += 1

    # if frameIndex >= 4000: # limits the execution to the first 4000 frames
    #    print("[INFO] cleaning up...")
    #    writer.release()
    #    vs.release()
    #    exit()

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
