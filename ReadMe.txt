Necessary packages to be installed:
  pip install argparse
  pip install open-python
  pip install numpy
  pip install imutils
  pip install math

Options and Arguments:
  -i < --image > : relative path to the image file
  -o < --objects > : number of objects to detect ( default value 1)
  -d < --detectionmethod > : number of detection method (default value 1)
  Example1:
    python detection-app.py -i filename.jpg -o 2 -d 2
  Example2:
    python detection-app.py --image filename.jpg -objects 2 -detectionmethod 2