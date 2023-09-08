# import the necessary packages
import numpy as np
import cv2
import imutils
import math

# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

def cal_poly_area(poly):
    s = 0
    n = len ( poly)
    for i in range ( 1, n - 1):
        j = i + 1
        x1 = poly[i][0] - poly[0][0]
        y1 = poly[i][1] - poly[0][1]
        x2 = poly[j][0] - poly[0][0]
        y2 = poly[j][1] - poly[0][1]
        s += (x1*y2 - x2*y1)/2
    return math.fabs(s)


def find_object_rec(img_path, object_number = 1, option = 1, color_mode = 1, perspective = True):

    #------------
    #-- main
    #------------

    # load the image, convert it to grayscale, and blur it
    #image = cv2.imread("example.jpg")
    image = cv2.imread(img_path)
    w0, h0 = image.shape[0], image.shape[1]
    # resize
    image = imutils.resize(image, width=700)
    image_copy = image
    w1, h1 = image_copy.shape[0], image_copy.shape[1]



    if option > 0:
        #contrast and brightness adjust
        auto_result, alpha, beta = automatic_brightness_and_contrast(image_copy)
        #print('alpha', alpha)
        #print('beta', beta)

        #cv2.imshow('auto_result', auto_result)
        #cv2.waitKey(0)

        image_copy = auto_result


    # detect edges in the image
    if color_mode == 1: #color mode
        cv2.imshow('auto_result', image_copy)
        cv2.waitKey(0)

        if option == 1:
            edged = cv2.Canny(image_copy, 15, 230) 
        elif option == 2:
            edged = cv2.Canny(image_copy, 30, 180) 
        elif option == 3:
            edged = cv2.Canny(image_copy, 45, 130)
        else:
            low_t = (int)(input('Please input low threshold: '))
            high_t = (int)(input('Please input high threshold: '))
            edged = cv2.Canny(image_copy, low_t, high_t) 
    else: #gray mode
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        cv2.imshow("Gray", gray)
        cv2.waitKey(0)

        if option == 1:
            edged = cv2.Canny(gray, 15, 230) 
        elif option == 2:
            edged = cv2.Canny(gray, 30, 180) 
        elif option == 3:
            edged = cv2.Canny(gray, 45, 130)
        else:
            low_t = (int)(input('Please input low threshold: '))
            high_t = (int)(input('Please input high threshold: '))
            edged = cv2.Canny(gray, low_t, high_t) 

    #----------------
    # ---------------# varying 2nd and 3rd parameter is helpful
    #------------------2nd term = 20 - 40
    #------------------3rd term = 50 - 200
    #----------------
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)

    # construct and apply a closing kernel to 'close' gaps between 'white'
    # pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    #----------------
    # ---------------# varying 2nd parameter is helpful, (7,7) (9,9)
    #----------------
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Closed", closed)
    cv2.waitKey(0)

    # define a (3, 3) structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.dilate(closed, None, iterations=2)
    #----------------
    # ---------------# varying 2nd and 3rd parameter is helpful
    #---------------- 2nd term = None & kernel
    #---------------- 3rd term = iteration 2 to 3
    #----------------
    cv2.imshow("Dilate", closed)
    cv2.waitKey(0)

    # find contours (i.e. the 'outlines') in the image and initialize the
    # total number of rectangles found
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = 0

    rect_contour_list = []
    rect_poly_list = []
    res_list = []
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if the approximated contour has four points, then assume that the
        # contour is a rectangle -- a rectangle and thus has four vertices
        if len(approx) == 4:
            poly = []
            for p in approx:
                poly.append ( (p[0][0], p[0][1]))
            
            rect_contour_list.append ( approx)
            rect_poly_list.append ( poly)
            

            total += 1

    #sort rectangle according to their area size
    for i in range ( total):
        for j in range ( i + 1, total):
            si = cal_poly_area ( rect_poly_list[i])
            sj = cal_poly_area ( rect_poly_list[j])
            if si < sj:
                rect_poly_list[i], rect_poly_list[j] = rect_poly_list[j], rect_poly_list[i]
                rect_contour_list[i], rect_contour_list[j] = rect_contour_list[j], rect_contour_list[i]
    
    whole_area = image_copy.shape[0]*image_copy.shape[1]
    cur_cnt = 0
    for i in range ( total):
        si = cal_poly_area ( rect_poly_list[i])
        if si/whole_area >= 0.9:
            continue
        if si/whole_area < 0.1:
            break
        cur_cnt+=1
        approx = []
        cur_rect = []
        
        for j in range ( 4):
            if perspective:
                x = rect_poly_list[i][j][0]
                y =rect_poly_list[i][j][1]                
                approx.append ( rect_contour_list[i][j])
                cur_rect.append ( (x*w0/w1, y*h0/h1))                
            else:
                k = ( j + 2 ) % 4
                xj = rect_poly_list[i][j][0]
                yj =rect_poly_list[i][j][1]
                xk = rect_poly_list[i][k][0]
                yk =rect_poly_list[i][k][1]
                round_r = 10
                x_dir = 1
                y_dir = 1
                if xk < xj:
                    x_dir = -1
                if yk < yj:
                    y_dir = -1
                x = xj + round_r*x_dir
                y = yj + round_r*y_dir
                rect_contour_list[i][j][0][0] = x
                rect_contour_list[i][j][0][1] = y

                approx.append ( rect_contour_list[i][j])
                cur_rect.append ( (x*w0/w1, y*h0/h1))                

        approx = rect_contour_list[i]
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
        
        res_list.append ( cur_rect)
        if cur_cnt >= object_number:
            break



    # display the output
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    return res_list

def main():
    for opt in range ( 1, 5):
        rect_list = find_object_rec("test_image.png", 2, opt, 1)
        print ( rect_list)

if __name__ == '__main__':
    main()