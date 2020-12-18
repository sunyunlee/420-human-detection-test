import sys
import cv2
from tqdm import tqdm

from sd_io import get_projection_parameters, \
    generate_image_output, generate_video_output
from sd_detect import detect_people
from sd_measure import measure_locations


def path_is_video(path):
    return path.lower().endswith(('.avi', '.mpg', '.mp4'))


def handleVideoFlow(videoPath, outputPath, detectMethod):
    """
    Runs the social detection flow on the video.

    Arguments:
        videoPath: Relative path to the video.
        outputPath: Relative path for saving the output.
        detectMethod: The human detection method used.
    """
    vidcap = cv2.VideoCapture(videoPath)
    frameCount = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))

    roi = None
    scale = None
    frame_seq = []
    people_seq = []

    print('Processing frames')
    # Process every 4th frame because minor movements shouldn't signficantly affect the output.
    for offset in tqdm(range(frameCount // 4)):
        frameNum = 4 * offset

        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
        success, frame = vidcap.read()

        if not success:
            vidcap.release()
            break

        if frameNum == 0:
            roi, scale = get_projection_parameters(frame)

        bboxes = detect_people(frame, detectMethod)
        people = measure_locations(frame, bboxes, roi, scale)
        frame_seq.append(frame)
        people_seq.append(people)

    generate_video_output(frame_seq, people_seq, fps // 4, outputPath) # Passig fps // 4 to slow down the output video.


def handleImageFlow(imagePath, outputPath, detectMethod):
    """
    Runs the social detection flow on the image.

    Arguments:
        imagePath: Relative path to image.
        outputPath: Relative path for saving the output.
        detectMethod: The human detection method used.
    """
    image = cv2.imread(imagePath)
    roi, scale = get_projection_parameters(image)
    bboxes = detect_people(image, detectMethod)
    people = measure_locations(image, bboxes, roi, scale)
    generate_image_output(image, people, outputPath)


def main():
    path = sys.argv[1]
    outputPath = sys.argv[2]
    if len(sys.argv) <= 3:
        detect_method = 'yolov3'
    else:
        detect_method = sys.argv[3]

    if path_is_video(path):
        # Input is a video
        handleVideoFlow(path, outputPath, detect_method)
    else:
        # Input is static image
        handleImageFlow(path, outputPath, detect_method)


# Usage: "python sd_main.py <path to input image> <path to output> [<detection method>]"
# <detection method> is 'yolov3' if not specified
if __name__ == "__main__":
    main()
