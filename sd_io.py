import numpy as np


def get_projection_parameters(image):
    """
    Gets the projection parameters (ROI and scale) for the specified image.

    This may involve prompting the user to input the parameters, or just reading
    them from a text file if that's ever easier for testing purposes.

    This function returns a tuple of two values.
    * The first value represents the ROI. It is a tuple of four values, each of
    which is a point (y, x) on the image that is a vertex of the ROI
    quadrilateral. The order of the vertices must be bottom-left, bottom-right,
    top-right, top-left.
    * The second value represents the horizontal and vertical scale. It is a
    tuple of three points (y, x) on the image. The first and second points are
    supposed to form a horizontal line two meters long in physical space. The
    first and third points are supposed to form a vertical line two meters tall
    in physical space.
    """
    return ((0, 0), (0, 0), (0, 0), (0, 0)), ((0, 0), (0, 0), (0, 0))


def generate_output(image, people):
    """
    Generates output based on the specified image and the people + social
    distancing information detected in it.

    This output might consist of the image with colored rectangles drawn around
    the people, for instance.

    <people> is a list of dicts representing the people in the image, formatted
    like the output of sd_measure.measure_locations().
    """
    pass
