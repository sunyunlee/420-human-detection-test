# Code adapted from https://github.com/deepak112/Social-Distancing-AI/tree/08a9a21ccf8ced3e6ff270628cb1c9b21a55fbee
import os
import time
import numpy as np
import cv2
import operator
import plotly.graph_objects as go


displayImage = None
mousePts = None


# Dimensions of bird's-eye view output videos
BEV_SIDE = 750


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

    
def generate_image_output(image, people, fileName):
    """
    Generates output based on the specified image and the people + social
    distancing information detected in it. It saves the output in the output directory using the given fileName.
    Namely, it saves the output as a file named fileName.jpg, a birds eye view of the output as a file named
    fileNameBirdsEye.jpg, and a point cloud output as a file named fileNamePointCloud.html.

    <people> is a list of dicts representing the people in the image, formatted
    like the output of sd_measure.measure_locations().
    """
    image = drawBoxesAndLines(image, people)

    # Do a first pass through the people information to find the boundaries of
    # the smallest rectangle that contains all their horizontal locations,
    # which we'll use to generate the bird's-eye view
    X1 = None
    Y1 = None
    X2 = None
    Y2 = None
    for j in range(len(people)):
        location = people[j]['location']
        X = location[0]
        Y = location[1]
        if X1 is None or X < X1:
            X1 = X
        if Y1 is None or Y < Y1:
            Y1 = Y
        if X2 is None or X < X2:
            X2 = X
        if Y2 is None or Y > Y2:
            Y2 = Y
    
    birdsEyeImage = generateBirdsEyeView(image, people, X1, Y1, X2, Y2)
    
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("output/" + fileName):
        os.makedirs("output/" + fileName)
    cv2.imwrite("output/" + fileName + "/detection.jpg", image)
    cv2.imwrite("output/" + fileName + "/birdseye.jpg", birdsEyeImage)
    plotPointCloud(people, "output/" + fileName + "/pointcloud.html")


def generate_video_output(frame_seq, people_seq, fps,
                          fileName):
    """
    Generates output based on the specified video and the people + social
    distancing information detected in it. It saves the output in the output
    directory using the given fileName. Namely, it saves the output as a file
    named fileName.avi, a birds eye view of the output as a file named
    fileNameBirdsEye.avi, and a point cloud output as a file named
    fileNamePointCloud.html.

    <frame_seq> is a list of the video's frames, in order, as individual images.

    <people_seq> is a list of the same length as <frame_seq>. Each element is
    itself a list of dicts representing the people in the corresponding frame,
    formatted like the output of sd_measure.measure_locations().
    """
    if not os.path.exists("output"):
        os.makedirs("output")
    if not os.path.exists("output/" + fileName):
        os.makedirs("output/" + fileName)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    output_movie = cv2.VideoWriter(
        "output/" + fileName + "/detection.avi", fourcc, fps,
        (frame_seq[0].shape[1], frame_seq[0].shape[0]))
    birdsEyeMovie = cv2.VideoWriter(
        "output/" + fileName + "/BirdsEye.avi",
        fourcc, fps, (BEV_SIDE, BEV_SIDE))

    # Do a first pass through the people information to find the boundaries of
    # the smallest rectangle that contains all their horizontal locations,
    # which we'll use to generate the bird's-eye view
    X1 = None
    Y1 = None
    X2 = None
    Y2 = None
    for i in range(len(people_seq)):
        for j in range(len(people_seq[i])):
            location = people_seq[i][j]['location']
            X = location[0]
            Y = location[1]
            if X1 is None or X < X1:
                X1 = X
            if Y1 is None or Y < Y1:
                Y1 = Y
            if X2 is None or X < X2:
                X1 = X
            if Y2 is None or Y > Y2:
                Y2 = Y
    
    # Now, render the videos
    for i in range(len(frame_seq)):
        frame = frame_seq[i]
        people = people_seq[i]

        frame = drawBoxesAndLines(frame, people)
        birdsEyeFrame = generateBirdsEyeView(frame, people, X1, Y1, X2, Y2)

        output_movie.write(frame)
        birdsEyeMovie.write(birdsEyeFrame)

    output_movie.release()
    birdsEyeMovie.release()

    plotPointCloud(people_seq[0], "output/" + fileName + "PointCloud.html")


def drawBoxesAndLines(image, people):
    """
    Draws the bounding boxes and lines on the given image.

    <people> is a list of dicts representing the people in the image, formatted
    like the output of sd_measure.measure_locations().
    """
    for d in people:
        topLeft = (d['bbox'][3], d['bbox'][2])
        bottomRight = (d['bbox'][1], d['bbox'][0])
        center = tuple(coord // 2 for coord in tuple(map(operator.add, topLeft, bottomRight)))
        # Draw bbox around person
        if len(d['too_close']) == 0:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        image = cv2.rectangle(image, topLeft, bottomRight, color, 2)

        for idx in d['too_close']:
            coordsOfOtherPerson = people[idx]['bbox']
            centerOfOtherPerson = ((coordsOfOtherPerson[1] + coordsOfOtherPerson[3]) // 2,
                                   (coordsOfOtherPerson[0] + coordsOfOtherPerson[2]) // 2)
            image = cv2.line(image, center, centerOfOtherPerson, (0, 0, 255), 2)
    return image


def plotPointCloud(people, output_path):
    """
    Plots a point cloud of <people>'s locations in 3D space, saving it at
    <output_path>.

    <people> is a list of dicts representing the people, formatted like the
    output of sd_measure.measure_locations().
    """
    x = []
    y = []
    z = []
    colors = []
    for p in people:
        location = p['location']
        x.append(location[0])
        y.append(location[1])
        z.append(location[2])
        if len(p['too_close']) == 0:
            colors.append('#00ff00')
        else:
            colors.append('#ff0000')

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
            opacity=1
        ))],
        layout=go.Layout(
            scene=dict(
                aspectmode='data'
            )))
    fig.write_html(file=output_path)


def generateBirdsEyeView(image, people, X1, Y1, X2, Y2):
    """
    Draws the bird's-eye representation of the given image.

    <people> is a list of dicts representing the people in the image, formatted
    like the output of sd_measure.measure_locations().
    """
    output = np.zeros((BEV_SIDE, BEV_SIDE, 3), np.uint8)

    red = []
    green = []

    scale = max(X2 - X1, Y2 - Y1)
    xoffset = (BEV_SIDE - (X2 - X1) / scale * BEV_SIDE)/2
    yoffset = (BEV_SIDE - (Y2 - Y1) / scale * BEV_SIDE)/2
    
    personScreenCoords = []
    for i in range(len(people)):
        location = people[i]['location']
        x = int((location[0] - X1) / scale * BEV_SIDE + xoffset)
        y = int((location[1] - Y1) / scale * BEV_SIDE + yoffset)
        personScreenCoords.append((x, y))
    
    for i1 in range(len(people)):
        too_close = people[i1]['too_close']
        coords = personScreenCoords[i1]
        if len(too_close) == 0:
            green.append(coords)
        else:
            red.append(coords)
            for i2 in too_close:
                output = cv2.line(output, coords, personScreenCoords[i2],
                                  (0, 0, 255), 2)

    for point in red:
        output = cv2.circle(output, point, 5, (0, 0, 255), 5)
    for point in green:
        output = cv2.circle(output, point, 5, (0, 255, 0), 5)

    return output
