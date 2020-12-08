import sys
import numpy as np
import cv2

from sd_io import get_projection_parameters, generate_output
from sd_detect import detect_people
from sd_measure import measure_locations


# Usage: "python sd_main.py <path to input image>"
if __name__ == "__main__":
    path = sys.argv[1]
    image = cv2.imread(path)
    roi, scale = get_projection_parameters(image)
    bboxes = detect_people(image)
    people = measure_locations(image, bboxes, roi, scale)
    generate_output(image, people)
