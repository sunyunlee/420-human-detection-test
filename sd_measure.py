import numpy as np


def measure_locations(image, bboxes, roi, scale):
    """
    Interprets the locations of people detected in the specified image as
    corresponding to locations in 3D space, and measures their distances to one
    another to check for social distancing.

    <bboxes> is the list of detected people's rectangular bounding boxes,
    formatted like the output of sd_detect.detect_people().

    <roi> and <scale> are the image's projection parameters, formatted like the
    output of sd_io.get_projection_parameters().

    This function returns a list of dicts. Each dict represents a person in the
    image, and has the following key-value pairs:
    * bbox: Represents the person's bounding box in the image. A tuple of four
    integers, formatted like in the output of sd_detect.detect_people().
    * location: Represents the location of the person's bottom center in 3D
    space. A tuple of three numbers (X, Y, Z). Scale is in meters.
    * too_close: A list containing the indices in the master list of the other
    people to whom this person is closer than two meters.
    """
    return []
