import numpy as np
import cv2


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
    * 'bbox': Represents the person's bounding box in the image. A tuple of four
    integers, formatted like in the output of sd_detect.detect_people().
    * 'location': Represents the location of the person's center in 3D space. A
    numpy array of three numbers [X, Y, Z], with Z as the vertical coordinate.
    Scale is in meters.
    * 'too_close': A list containing the indices in the master list of the other
    people to whom this person is closer than two meters.
    """

    transform = cv2.getPerspectiveTransform(
        np.float32(roi), np.float32(((1, 0), (1, 1), (0, 1), (0, 0))))
    tfed_scale = cv2.perspectiveTransform(np.float32((scale,)), transform)[0]
    two_meters = np.linalg.norm(tfed_scale[1] - tfed_scale[0])
    two_meters_v = np.linalg.norm(tfed_scale[2] - tfed_scale[0])
    ratio = two_meters_v / two_meters

    people = []
    for bbox in bboxes:
        #For now, we're treating everyone's center as the 3D point corresponding
        #to their 2D bounding box's bottom center
        center_2d = ((bbox[1] + bbox[3])/2, bbox[0])
        tfed_center_2d = cv2.perspectiveTransform(np.float32(((center_2d,),)),
                                                  transform)[0, 0]
        center_3d = np.array((tfed_center_2d[1]/ratio, tfed_center_2d[0], 0))
        people.append({'bbox': bbox, 'location': center_3d, 'too_close': []})

    for i1 in range(len(people)):
        p1 = people[i1]
        for i2 in range(i1):
            p2 = people[i2]
            dist = np.linalg.norm(p2['location'] - p1['location'])
            if dist < two_meters:
                p1['too_close'].append(i2)
                p2['too_close'].append(i1)

    for i in range(len(people)):
        print(str(i) + ': ' + str(people[i]))
    
    return people
