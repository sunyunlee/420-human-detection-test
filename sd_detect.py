import numpy as np
from typing import Union, Tuple, List
import cv2
from imutils.object_detection import non_max_suppression
import imutils
import matplotlib.pyplot as plt


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
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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


def HOG(img, win_stride: Tuple[int], padding: Tuple[int], scale: int):
    """
    Returns bounding boxes around humans that the HOG detectors detect
    * x1 is the person's bounding box's left
    * y1 is the person's bounding box's top 
    * x2 is the person's bounding box's right
    * y2 is the person's bounding box's bottom 
    """
    hog = cv2.HOGDescriptor()
    # hog = cv2.HOGDescriptor(img.shape, (8, 8), (4, 4), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    img = imutils.resize(img, width=int(scale*img.shape[1]))
    hog_params = {'winStride': win_stride, 'padding': padding, 'scale': 1.05}

    rects, weights = hog.detectMultiScale(img, **hog_params)

    # pick = non_max_suppression(rects, probs=None, overlapThresh=0.1)
    results = []
    for i, (x, y, w, h) in enumerate(rects): 
        x1 = int(x / scale)
        y1 = int(y / scale)
        x2 = int((x + w) / scale)
        y2 = int((y + h) / scale)
        results.append([x1, y1, x2, y2])

    return results, list(weights.flatten())


def remove_overlapping_boxes(boxes: List[List[int]]) -> List[List[int]]:
    """ Removes bounding boxes which completely overlaps another bounding box

    :param boxes: the bounding boxes
    :return: the filtered bounding boxes 
    """
    box_to_remove = []
    for i in range(len(boxes)):
        curr_box = boxes[i]
        for j in range(len(boxes)):
            if i != j: 
                new_box = boxes[j]
                if curr_box[0] <= new_box[0] and \
                    curr_box[1] <= new_box[1] and \
                        curr_box[2] >= new_box[2] and \
                            curr_box[3] >= new_box[3]:
                    box_to_remove.append(i)

    box_to_remove = sorted(list(set(box_to_remove)))

    for i in range(len(box_to_remove) - 1, -1, -1):
        boxes.pop(box_to_remove[i])

    return boxes
                

def HOG_remove_low_confidence(boxes: List[List[int]], weights: List[int], confid_thresh: int = 1):
    """ Returns bounding boxes and the associated weights with weights lower than 
    the confidence threshold 

    :param boxes: the bounding boxes 
    :param weights: the confidences/weights of the predictions
    :param confid_thresh: the confidence threshold
    :return: list of bounding boxes and the associated weights 
    """
    indices = []
    for i in range(len(weights)):
        if weights[i] < confid_thresh: 
            indices.append(i)
    
    boxes_filtered = remove_indices(boxes, indices)
    weights_filtered = remove_indices(weights, indices)

    return boxes_filtered, weights_filtered


def remove_indices(l: list, indices: List[int]) -> list:
    """ Given indices and a list, the function returns a list with the items at indices removed

    :param l: the list of items
    :param indices: list of indices at which to remove the items
    :return: list l with items removed 
    """
    for i in range(len(indices) - 1, -1, -1):
        if indices[i] >= len(l):
            raise Exception("Index out of bounds!")
        l.pop(indices[i])
    
    return l
