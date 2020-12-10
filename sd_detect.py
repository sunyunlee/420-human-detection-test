import numpy as np
from typing import Union 
import cv2
from imutils.object_detection import non_max_suppression
import imutils


weight_path = "yolov3.weights"
config_path = "yolov3.cfg"


def detect_people(image, method: str = Union["yolov3", "hog"]):
    """
    Identifies the rectangular areas where people appear in the specified image.

    This function returns a list. Each element in the list represents a detected
    person, and is a tuple of four integers. In order, these integers are:
    * y-coordinate on the image of the person's bounding box's bottom
    * x-coordinate on the image of the person's bounding box's right
    * y-coordinate on the image of the person's bounding box's top
    * x-coordinate on the image of the person's bounding box's left
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "yolov3":
        result = YOLOv3(image_gray)
    # elif method == "hog":
    #     result = HOG(image)

    return result


def YOLOv3(img):
    """
    Identifies the rectangular areas using yolov3 pre trained model 

    Ref: https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
    """
    # Create a model from pre-trained weights 
    model = cv2.dnn.readNetFromDarknet(config_path, weight_path)

    # Change the image size as the input yolo size (416, 416)
    input_image = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Determine the output layer
    layer_names = model.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]

    # Forward pass
    model.setInput(input_image)
    outputs = model.forward(layer_names)

    # # Get bounding boxes 
    boxes_filtered = process_yolov3_output(outputs, img.shape[:2])

    return boxes_filtered


def process_yolov3_output(outputs, input_shape):
    """
    From the output of the model, select detections for humans, and which confidence 
    score greater than 0.5
    The output is in the format [x1, y1, x2, y2]
    * x1 is the person's bounding box's left
    * y1 is the person's bounding box's top 
    * x2 is the person's bounding box's right
    * y2 is the person's bounding box's bottom 

    Ref: https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
    """    
    boxes = []
    confid_scores = []
    confid_thresh = 0.5
    thresh = 0.5
    H, W = input_shape[0], input_shape[1]

    for output in outputs:
        for pred in output: # first four indices indicate the location, rest are the prediction scores 
            scores = pred[5:]
            class_id = np.argmax(scores)
            confid_score = scores[class_id]
            
            if class_id == 0: # human class index in COCO
                if confid_score > confid_thresh: # need to be confident in prediction
                    box = pred[0:4] * np.array([W, H, W, H])
                    (x_center, y_center, width, height) = box.astype("int")
                    
                    # Get top left corner pixel coordinates 
                    x = int(x_center - (width / 2))
                    y = int(y_center - (width / 2))
                    
                    boxes.append([x, y, int(width), int(height)])
                    confid_scores.append(float(confid_score))

    # Non maximum suppression to remove any overlapping boxes               
    indices = cv2.dnn.NMSBoxes(boxes, confid_scores, confid_thresh, thresh)
    indices = indices.flatten()
    boxes_filtered = []

    for i in indices:
        x,y,w,h = boxes[i]
        left, top, right, bottom = x, y, x + w, y + h
        boxes_filtered.append([left, top, right, bottom])

    return boxes_filtered


def HOG(img, scale: int=0.5):
    """
    Returns bounding boxes around humans that the HOG detectors detect
    * y-coordinate on the image of the person's bounding box's bottom
    * x-coordinate on the image of the person's bounding box's right
    * y-coordinate on the image of the person's bounding box's top
    * x-coordinate on the image of the person's bounding box's left 
    """
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    scale = 0.5

    img = imutils.resize(img, width=int(scale*img.shape[1]))
    rects, weights = hog.detectMultiScale(img, winStride=(2, 2), padding=(10, 10), scale=scale)

    # pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    results = []
    for i, (x, y, w, h) in enumerate(rects): 
        if weights[i][0] > 0.2:
            x1 = int(x / scale)
            y1 = int(y / scale)
            x2 = int((x + w) / scale)
            y2 = int((y + h) / scale)
            results.append([x1, y1, x2, y2])
            
    return results
