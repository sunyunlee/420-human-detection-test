import sys
import numpy as np
import cv2

from sd_io import get_projection_parameters,\
     generate_image_output, generate_video_output
from sd_detect import detect_people
from sd_measure import measure_locations


def path_is_video(path):
    return path.lower().endswith(('.avi', '.mpg', '.mp4'))


# Usage: "python sd_main.py <path to input image> [<detection method>]"
# <detection method> is 'yolov3' if not specified
if __name__ == "__main__":
    path = sys.argv[1]
    if len(sys.argv) <= 2:
        detect_method = 'yolov3'
    else:
        detect_method = sys.argv[2]
    
    if path_is_video(path):
        # Input is video; process it by processing each frame individually
        vidcap = cv2.VideoCapture(path)
        roi = None
        scale = None
        frame_seq = []
        people_seq = []
        frame_num = 0
        success = True
        while success:
            print('Processing frame %d' % frame_num)
            success, frame = vidcap.read()
            if frame_num == 0:
                roi, scale = get_projection_parameters(frame)
            bboxes = detect_people(frame, detect_method)
            people = measure_locations(frame, bboxes, roi, scale)
            frame_seq.append(frame)
            people_seq.append(people)
            frame_num += 1
        generate_video_output(frame_seq, people_seq)
    else:
        # Input is static image
        image = cv2.imread(path)
        roi, scale = get_projection_parameters(image)
        bboxes = detect_people(image, detect_method)
        people = measure_locations(image, bboxes, roi, scale)
        generate_image_output(image, people)
