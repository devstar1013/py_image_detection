import argparse
import cv2
from find_contour import find_object_rec
from transform import four_point_transform
import numpy as np
import imutils
import os


def showHelp():
    print ( 'Options and Arguments:')
    print ( ' -i < --image > : relative path to the image file')
    print ( ' -o < --objects > : number of objects to detect ( default value 1)')
    print ( ' -d < --detectionmethod > : number of detection method (default value 1)')
    print ( 'Example1:')
    print ( '  python detection-app.py -i filename.jpg -o 2 -d 2')
    print ( 'Example2:')
    print ( '  python detection-app.py --image filename.jpg -objects 2 -detectionmethod 2')
    print()
def main():
    # python detection-app.py --image filename.jpg --objects 2 --detectionmethod 2
    # get the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help = "path to the image file")
    ap.add_argument("-o", "--objects",help = "number of object to detect")
    ap.add_argument("-d", "--detectionmethod",help = "number of detection method")
    ap.add_argument("-c", "--colormode",help = "1 - color detection, 0 - gray detection")
    ap.add_argument("-p", "--perspective",help = "on - perspective transform, off - disable perspective transform")
    args = vars(ap.parse_args())
    
    if args['image'] == None:
        print ( 'Command Error:')
        showHelp()
        return
    if args['objects'] == None:
        args['objects'] = 1 # default 1 object to detect
    if args['detectionmethod'] == None:
        args['detectionmethod'] = 1 # default first detection method
    if args['colormode'] == None:
        args['colormode']  = 1 # default color detection
    if args['perspective'] == None:
        args['perspective']  = "on" # default color detection        

    perspective = (args['perspective'] == "on")
    #detect object
    rect_list = find_object_rec(args['image'], (int)(args['objects']), (int)(args['detectionmethod']), (int)(args['colormode']), perspective)

    #rotate
    image = cv2.imread(args["image"])
    #image = imutils.resize(image, width=700)

    id = 0
    if len ( rect_list) < (int)(args['objects']):
        print ( 'Failed to find all objects: Missing %d object' % ((int)(args['objects']) - len ( rect_list)))
    
    if not os.path.exists("result"):
        os.mkdir ( "result")

    for coord in rect_list:
        pts = np.array(coord, dtype = "float32")
        warped = four_point_transform(image, pts)
        id+=1
        #user option mode 
        while True:
            warped_resize = imutils.resize(warped, width=700)
            cv2.imshow("Warped", warped_resize)
            cv2.waitKey(0)
            rotate_ans = input("Do you want to rotate the cropped image (y/n) ? ")
            if rotate_ans == 'y':
                warped = imutils.rotate_bound ( warped, -90)
            else:
                break
        file_name = './result/result_%d.jpg' % (id)
        cv2.imwrite(file_name, warped)
if __name__ == '__main__':
    main()