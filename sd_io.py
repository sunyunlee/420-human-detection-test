# Code mainly taken from https://github.com/deepak112/Social-Distancing-AI/tree/08a9a21ccf8ced3e6ff270628cb1c9b21a55fbee
import time
import numpy as np
import cv2
import operator

displayImage = None
mousePts = None


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
    global displayImage, mousePts
    displayImage = image.copy()
    mousePts = []

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", getMousePts)

    while True:
        cv2.imshow("image", displayImage)
        cv2.waitKey(1)
        if len(mousePts) == 8:
            break

    # Should destroy the window, but it's not working.
    cv2.destroyWindow("image")
    cv2.waitKey(1)

    return ((mousePts[0], mousePts[1], mousePts[2], mousePts[3]),
            (mousePts[4], mousePts[5], mousePts[6]))


def getMousePts(event, x, y, flags, param):
    """
    Handles mouse click events by adding the (x, y) coordinate to the global
    mousePts list and drawing the circle on the image.

    https://docs.opencv.org/master/d7/dfc/group__highgui.html#gab7aed186e151d5222ef97192912127a4

    Arguments:
        event: The mouse click event.
        x: the x-coordinate of the point on the image that was clicked.
        y: the y-coordinate of the point on the image that was clicked.
        flags: one of the cv::MouseEventFlags constants.
        param: an optional parameter.
    """
    global displayImage, mousePts

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mousePts) < 4:
            cv2.circle(displayImage, (x, y), 10, (0, 255, 0), -1)
        else:
            cv2.circle(displayImage, (x, y), 10, (255, 0, 0), -1)

        if 1 <= len(mousePts) <= 3:
            # Draw a line connecting the new point with the most recently added
            # point.
            cv2.line(displayImage, (x, y), (mousePts[-1][1], mousePts[-1][0]),
                     (70, 70, 70),
                     2)
            if len(mousePts) == 3:
                # Draw a line connecting the final point with the first point to
                # close off the rectangle.
                cv2.line(displayImage, (x, y), (mousePts[0][1], mousePts[0][0]),
                         (70, 70, 70), 2)

        if len(mousePts) >= 5:
            cv2.line(displayImage, (x, y), (mousePts[4][1], mousePts[4][0]),
                     (70, 70, 70), 2)

        mousePts.append((y, x))


def generate_image_output(image, people, outputPath):
    """
    Generates output based on the specified image and the people + social
    distancing information detected in it. It saves the generated image at the given outputPath.

    <people> is a list of dicts representing the people in the image, formatted
    like the output of sd_measure.measure_locations().
    """
    image = drawBoxesAndLines(image, people)
    cv2.imwrite(outputPath, image)


def generate_video_output(frame_seq, people_seq, fps, outputPath):
    """
    Generates output based on the specified video and the people + social
    distancing information detected in it. It saves the generated video at the given outputPath.

    <frame_seq> is a list of the video's frames, in order, as individual images.

    <people_seq> is a list of the same length as <frame_seq>. Each element is
    itself a list of dicts representing the people in the corresponding frame,
    formatted like the output of sd_measure.measure_locations().
    """
    if not frame_seq:
        return

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    output_movie = cv2.VideoWriter(outputPath, fourcc, fps, (frame_seq[0].shape[1], frame_seq[0].shape[0]))

    numberOfFrames = len(frame_seq)
    for i in range(numberOfFrames):
        frame = frame_seq[i]
        people = people_seq[i]
        frame = drawBoxesAndLines(frame, people)
        output_movie.write(frame)

    output_movie.release()


def drawBoxesAndLines(image, people):
    """
    Draws the bounding boxes and lines on the given image.

    <people> is a list of dicts representing the people in the image, formatted
    like the output of sd_measure.measure_locations().
    """
    for d in people:
        topLeft = (d['bbox'][2], d['bbox'][3])
        bottomRight = (d['bbox'][0], d['bbox'][1])
        center = tuple(coord // 2 for coord in tuple(map(operator.add, topLeft, bottomRight)))
        # Draw bbox around person
        image = cv2.rectangle(image, topLeft, bottomRight, (0, 0, 255), 2)

        for idx in d['too_close']:
            coordsOfOtherPerson = people[idx]['bbox']
            centerOfOtherPerson = ((coordsOfOtherPerson[2] + coordsOfOtherPerson[0]) // 2,
                                   (coordsOfOtherPerson[3] + coordsOfOtherPerson[1]) // 2)
            image = cv2.line(image, center, centerOfOtherPerson, (0, 0, 255), 2)
    return image
