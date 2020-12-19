# CSC420 Social Distancing Detection

## Usage:
* Run following command:
    * python sd_main.py \<path to input> \<output name> \[\<detection method>]
    * \<output name> will be the name of a folder within the output/ folder where the output will be placed.
    * \<detection method> is an optional parameter which could either be 'hog' or 'yolov3'. It defaults to 'yolov3' if not specified 
* After running the command above, you will be prompted with an image on which you need to draw 7 points. The first 4 points will define a rectangle around the region of interest. This process is order sensitive, so the first point should be the bottom left corner of the rectangle if seen from above, the second point is the bottom right corner, third is the top right and fourth is the top left. After that, you should specify 3 points: The first 2 should form a horizontal line, parallel to the rectangle, that is measures 6 feet in length using the scale of the image. The first and 3rd point from a vertical line, parallel to the rectangle, that measures 6 feet in length using the scale of the image.   
