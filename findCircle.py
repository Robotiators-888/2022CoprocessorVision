import imp
from pickle import TRUE
import cv2
import numpy as np
import time
import json
import math
import sys

import math
import struct
import socket

global sock

DEBUG = False
NOGUI = False

img_path = None

# get command line arguments
if len(sys.argv) > 1:
    print(len(sys.argv))
    arg2 = sys.argv[1]
    if arg2 == '-h' or arg2 == '-help':
        print('usage: python3 findCircle2.py [-debug|-release|-nogui]')
        print('-debug: runs in debug mode, with gui sliders to adjust parameters and displays images')
        print('-release: runs without a gui, sends data over UDP to your robot')
        print('-nogui: runs without a gui in debug mode')
        print('')
        print('use image/video as input:')
        print('-i <path>: path to image file')
        print('-v <path>: path to video file')
        print('')
        print('Gui keybinds/commands:')
        print('s: print python object to console with slider values')
        print('q: quit')
        print('o: save current image to file as frame.png')
        exit()
    elif (arg2 == '-debug'):
        print('Debug mode')
        DEBUG = True
        NOGUI = False
    elif (arg2 == '-release'):
        print('Release mode')
        DEBUG = False
        NOGUI = True
    elif (arg2 == '-nogui'):
        print('No GUI mode')
        DEBUG = True
        NOGUI = True
    else:
        print('missing argument')
        print('usage: python3 findCircle2.py [-debug|-release|-nogui]')
        print('Try -help for more info')
        exit()
    if (len(sys.argv) > 3):
        arg3 = sys.argv[2]
        # allow using a image path as input instead of camera
        if (arg3 == '-i'):
            print('Using image as input')
            arg4 = sys.argv[3]
            img_path = arg4
else:
    print('missing argument')
    print('usage: python3 findCircle2.py [-debug|-release|-nogui]')
    print('Try -help for more info')
    exit()


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 8888))


def sendUDP(x, y, detected):
    global sock
    msg = struct.pack(">ff?", x, y, detected)
    sock.sendto(msg, ("127.0.0.1", 5802))

    return "Success"


def Arrhenius(x, a, b):
    e = math.e
    p1 = b/x
    return a*e**p1


def Caunchy(x, a, b, c):
    y = a + (b/x**2) + (c/x**4)
    return y


def Exponential(x, a, b, c):
    y = a*(math.e**(b*x)) + c
    return y


def isInRange(X, X2, t):
    y = Arrhenius(X, A, B)
    if abs(X-X2) < t:
        return True
    else:
        return False

# TODO: use radius of circle to find new min max for Hough Alg Size
# then withing while loop run Hough circles again to get super accurate radius


# background = image_raw
# background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
#img_path = 'ballr/redbig_2.mp4/img/5.png'


def nothing(x):
    pass


def inchesToMeters(inches):
    return inches * 0.0254


def getDistance(pr):
    fl = 540.0  # new
    r = 4.5
    d = ((fl*r)/pr)#*1.008
    # use Hconstant
    CONSTANT = 2500
    c = CONSTANT/pr
    try:
        xdist = math.sqrt((c**2)-(d**2))
        ydist = d
        return [xdist, ydist]
    except ValueError:
        xdist = 0
        ydist = 0
    #print('xdist', xdist,'ydist', ydist)
        # p = i[2]
for i in range(85,90): print(getDistance(i))

# function to check if the inputed meters away Y corridinate corilates with how close the ball should be to the horrizen based on inputed pixel radius
# def horizen(px_radius,y_meters_away):
valueTracker = []
# if not (NOGUI):
if (NOGUI == False):
    cv2.namedWindow('slider')
    cv2.namedWindow('slider2')
    cv2.createTrackbar('hue', 'slider', 0, 179, nothing)  # RED BALL VALUES
    cv2.createTrackbar('sat', 'slider', 81, 255, nothing)
    cv2.createTrackbar('val', 'slider', 65, 255, nothing)

    # make second sliders for uppers
    cv2.namedWindow('slider2')
    cv2.createTrackbar('hue2', 'slider2', 12, 179, nothing)
    cv2.createTrackbar('sat2', 'slider2', 255, 255, nothing)
    cv2.createTrackbar('val2', 'slider2', 255, 255, nothing)

    cv2.createTrackbar('hue', 'slider', 0, 179, nothing)
    valueTracker.append(('hue', 'slider'))  # ORANGE BALL VALUES
    cv2.createTrackbar('sat', 'slider', 81, 255, nothing)
    valueTracker.append(('sat', 'slider'))
    cv2.createTrackbar('val', 'slider', 65, 255, nothing)
    valueTracker.append(('val', 'slider'))

    # make second sliders for uppers

    cv2.createTrackbar('hue2', 'slider2', 12, 179, nothing)
    valueTracker.append(('hue2', 'slider2'))
    cv2.createTrackbar('sat2', 'slider2', 255, 255, nothing)
    valueTracker.append(('sat2', 'slider2'))
    cv2.createTrackbar('val2', 'slider2', 255, 255, nothing)
    valueTracker.append(('val2', 'slider2'))

    # asjuster for cv2.Canny
    cv2.createTrackbar('th1', 'slider2', 119, 255, nothing)
    valueTracker.append(('th1', 'slider2'))
    cv2.createTrackbar('th2', 'slider2', 53, 255, nothing)
    valueTracker.append(('th2', 'slider2'))
    # ajuster for cv2.HoughCircles
    cv2.createTrackbar('minRadius', 'slider2', 0, 150, nothing)
    valueTracker.append(('minRadius', 'slider2'))
    cv2.createTrackbar('maxRadius', 'slider2', 1, 150, nothing)
    valueTracker.append(('maxRadius', 'slider2'))
    cv2.createTrackbar('parem1', 'slider2', 15, 100, nothing)
    valueTracker.append(('parem1', 'slider2'))
    cv2.createTrackbar('parem2', 'slider2', 30, 100, nothing)
    valueTracker.append(('parem2', 'slider2'))
    cv2.createTrackbar('minDist', 'slider2', 1, 100, nothing)
    valueTracker.append(('minDist', 'slider2'))
    cv2.createTrackbar('maxDist', 'slider2', 20, 100, nothing)
    valueTracker.append(('maxDist', 'slider2'))
    cv2.createTrackbar('HConstant', 'slider2', 2000, 3000, nothing)
    valueTracker.append(('HConstant', 'slider2'))


OrangeBallValues = {('hue', 'slider'): 0, ('sat', 'slider'): 81, ('val', 'slider'): 65, ('hue2', 'slider2'): 12, ('sat2', 'slider2'): 255, ('val2', 'slider2'): 255,
                    ('th1', 'slider2'): 119, ('th2', 'slider2'): 53, ('minRadius', 'slider2'): 52, ('maxRadius', 'slider2'): 100, ('parem1', 'slider2'): 15, ('parem2', 'slider2'): 26}
RedTrainingValues = {('hue', 'slider'): 0, ('sat', 'slider'): 71, ('val', 'slider'): 13, ('hue2', 'slider2'): 16, ('sat2', 'slider2'): 255, ('val2', 'slider2'): 255, ('th1', 'slider2'): 184, ('th2', 'slider2')
                      : 53, ('minRadius', 'slider2'): 30, ('maxRadius', 'slider2'): 200, ('parem1', 'slider2'): 19, ('parem2', 'slider2'): 21, ('minDist', 'slider2'): 1, ('maxDist', 'slider2'): 100, ('HConstant', 'slider2'): 2000}
BlueTrainingValues = {('hue', 'slider'): 32, ('sat', 'slider'): 71, ('val', 'slider'): 13, ('hue2', 'slider2'): 146, ('sat2', 'slider2'): 255, ('val2', 'slider2'): 255, ('th1', 'slider2'): 184, ('th2', 'slider2')
                       : 53, ('minRadius', 'slider2'): 24, ('maxRadius', 'slider2'): 150, ('parem1', 'slider2'): 19, ('parem2', 'slider2'): 45, ('minDist', 'slider2'): 1, ('maxDist', 'slider2'): 100, ('HConstant', 'slider2'): 2000}
# current ^

averageYValues = []
averageXValues = []
averageXQueueLength = 22
averageYQueueLength = 22
global averageCounter
averageCounter = 0

ValueMap = {}

# ROBOT VALUES SECTION:
#RedTrainingValues = {('hue', 'slider'): 0, ('sat', 'slider'): 74, ('val', 'slider'): 49, ('hue2', 'slider2'): 12, ('sat2', 'slider2'): 255, ('val2', 'slider2'): 255, ('th1', 'slider2'): 112, ('th2', 'slider2'): 53, ('minRadius', 'slider2'): 0, ('maxRadius', 'slider2'): 150, ('parem1', 'slider2'): 19, ('parem2', 'slider2'): 27, ('minDist', 'slider2'): 1, ('maxDist', 'slider2'): 100}
#RedTrainingValues = {('hue', 'slider'): 0, ('sat', 'slider'): 156, ('val', 'slider'): 73, ('hue2', 'slider2'): 179, ('sat2', 'slider2'): 255, ('val2', 'slider2'): 255, ('th1', 'slider2'): 112, ('th2', 'slider2'): 53, ('minRadius', 'slider2'): 30, ('maxRadius', 'slider2'): 150, ('parem1', 'slider2'): 19, ('parem2', 'slider2'): 14, ('minDist', 'slider2'): 1, ('maxDist', 'slider2'): 100, ('HConstant', 'slider2'): 2000}
for i in RedTrainingValues:
    ValueMap[(i[0], i[1])] = RedTrainingValues[i]

camera = cv2.VideoCapture(0)
# if (img_path == None):
#     try:
#         camera = cv2.VideoCapture(0)
#     except Exception as e:
#         print(e)
#         print("No camera found")
#         exit()


def findCircle(img):
    frame = img

    lower_color = np.array([ValueMap[('hue', 'slider')], ValueMap[(
        'sat', 'slider')], ValueMap[('val', 'slider')]])
    upper_color = np.array([ValueMap[('hue2', 'slider2')], ValueMap[(
        'sat2', 'slider2')], ValueMap[('val2', 'slider2')]])
    nothing = 0

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, None, iterations=4)
    mask = cv2.dilate(mask, None, iterations=4)
    #  shadowRectangle = empty frame
    shadowRectangle = np.zeros(frame.shape, np.uint8)
    orginalFrame = frame.copy()

    masked_image = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    # gray = gray-background

    # detect average location of brightest pixels in grey
    # and use that as the center of the circle
    averageLocation = np.where(gray == np.amax(gray))
    x = averageLocation[1][0]
    y = averageLocation[0][0]
    cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

    # remove small objects
    # gray = cv2.erode(gray, None, iterations=4)
    # gray = cv2.dilate(gray, None, iterations=4)
    #grey = cv2.medianBlur(gray, 5)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(
        gray, ValueMap['th1', 'slider2'], ValueMap['th2', 'slider2'])

    contour, hierarchy = cv2.findContours(edged,
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = cv2.findContours(
        gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # check if edged is completley black
    countour_image = np.zeros(
        (frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(countour_image, contour, -1, (0, 255, 0), 1)
    countour_image = cv2.cvtColor(countour_image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('gray', gray)

    if len(contour) != 0:
        circles = cv2.HoughCircles(gray,  cv2.HOUGH_GRADIENT,
                                   ValueMap[('minDist', 'slider2')],
                                   ValueMap[('maxDist', 'slider2')],
                                   param1=ValueMap[('parem1', 'slider2')],
                                   param2=ValueMap[('parem2', 'slider2')],
                                   minRadius=ValueMap[(
                                       'minRadius', 'slider2')],
                                   maxRadius=ValueMap[('maxRadius', 'slider2')])

        if circles is not None:
            #print("found circles!")
            largestDis = 0
            largestRadius = 0
            closestCircle = None
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2], (162, 255, 0), 2)
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
                # draw on maske_image
                cv2.circle(masked_image, (i[0], i[1]), i[2], (162, 255, 0), 2)
                cv2.circle(masked_image, (i[0], i[1]), 2, (0, 0, 255), 3)
                r = i[2]
                # check if radius is on left or right of screen
                # find ditstance from center of circle to bottom of screen
                distanceToBottom = frame.shape[0] - i[1] - i[2]
                # print('distanceToBottom',distanceToBottom)
                A = 163.5
                B = -879.9
                C = 1020.0

                try:
                    dis = None
                    if (i[0] < frame.shape[1]/2):
                        dis = getDistance(r)
                    else:
                        dis = getDistance(r)
                    # if (len(averageXValues) > 19):
                    #     averageX = sum(averageXValues)/len(averageXValues)
                    #     if (dis[0]/averageX > 1.4 or dis[0]/averageX < 0.6):
                    #         break
                    # print(len(averageXValues))
                    # if (len(averageXValues) > 19):
                    #     averageY = sum(averageYValues)/len(averageYValues)
                    #     print(dis[1]/averageY)
                    #     if (dis[1]/averageY > 1.4 or dis[1]/averageY < 0.6):
                    #         break

                    # TODO: SHADOW DETECTION
                    center = [i[0], i[1]]
                    center[1] = center[1] + int(i[2]*1.2)
                    # plot center on frame
                    cv2.circle(frame, center, 2, (255, 0, 0), 3)
                    # draw a rectange at center
                    #cv2.rectangle(frame, (center[0] - int(i[2]/2), center[1] - int(i[2]/7)), (center[0] + int(i[2]/2), center[1] + int(i[2]/7)), (0, 255, 0), 2)
                    # cut this rectangle from frame and save it as a shadow
                    shadowRectangle = frame[center[1] - int(i[2]/7):center[1] + int(
                        i[2]/7), center[0] - int(i[2]/2):center[0] + int(i[2]/2)]
                    # averag3e the rectangle
                    averageColor = 0
                    for x in range(shadowRectangle.shape[0]):
                        for y in range(shadowRectangle.shape[1]):
                            averageColor += shadowRectangle[x][y][2]
                    averageColor = averageColor / \
                        (shadowRectangle.shape[0]*shadowRectangle.shape[1])
                    print('averageColor', averageColor)
                    # was 40 but changed to 100
                    if averageColor > 50:
                        break

                    X = dis[1]/12
                    Y = Caunchy(X, A, B, C)
                    onGroundRatio = distanceToBottom/Y
                    if (onGroundRatio > -1 and onGroundRatio < 2.3):
                        # print('onGround',onGroundRatio)
                        totalDis = abs(dis[0]) + dis[1]
                        if (r > largestRadius):
                            largestDis = totalDis
                            largestRadius = r
                            closestCircle = i

                    else:
                        print('notonGround', onGroundRatio)

                    # get a rectange under the cricle from frame
                    # get the average of the rectangle

                    # create a image from the
                    # boundingRect = frame[

                    # convert to hsv

                    #average = cv2.cvtColor(averagePerColumn, cv2.COLOR_BGR2HSV)

                    # determine the dominant color from average

                    # draw the rectangle not touching the circle
                    #cv2.rectangle(frame, (center[0] - i[2], center[1] - i[2]), (center[0] + i[2], center[1] + int(i[2]/7), (0, 255, 0), 2))

                    #shadowRectangle = frame[i[1]-i[2]:i[1]+i[2], i[0]-i[2]:i[0]+i[2]]

                except Exception as e:
                    print("error finding ball", e)
            if (closestCircle is not None): # code that runs on the closest found ball
                i = closestCircle
                r = i[2]
                print('radius', i[2])
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 0, 255), 7)
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
                # draw on maske_image
                # label the ball on frame above the ball
                cv2.putText(
                    frame, 'Real Ball', (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.circle(masked_image, (i[0], i[1]), i[2], (0, 0, 255), 7)
                cv2.circle(masked_image, (i[0], i[1]), 2, (0, 0, 0), 3)

                dis = None
                if (i[0] < frame.shape[1]/2):
                    dis = getDistance(r)
                    dis[0] = -dis[0]
                else:
                    dis = getDistance(r)

                ydist = dis[1]
                xdist = dis[0]
                #ydist = ydist*2
                averageYValues.append(ydist)
                averageXValues.append(xdist)
                # remove first value
                if len(averageYValues) > averageYQueueLength:
                    averageYValues.pop(0)
                if len(averageXValues) > averageXQueueLength:
                    averageXValues.pop(0)
                # get average
                averageY = sum(averageYValues)/len(averageYValues)
                averageX = sum(averageXValues)/len(averageXValues)

                # TODO: tune output like this to desired position

                # send data over UDP to RoboRio
               #
                if (DEBUG == False):
                    sendUDP(averageX, averageY, True)

                # print('averageY',averageY,'averageX',averageX)
                # draw text in the corner of the screen
                roundedAverageY = round(averageY, 2)
                roundedAverageX = round(averageX, 2)
                # draw black rectange in border to hold text
                cv2.rectangle(
                    frame, (0, 0), (frame.shape[1], 75), (0, 0, 0), -1)
                cv2.putText(frame, 'Y: ' + str(roundedAverageY)+'in',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 230, 230), 2)
                cv2.putText(frame, 'X: ' + str(roundedAverageX)+'in',
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 230, 230), 2)
                global averageCounter
                if averageCounter < 0:
                    averageCounter -= 1
        else:
            if (DEBUG == False):
                sendUDP(0, 0, False)
            # averageCounter += 1
            # if averageCounter > 15:
            #     averageCounter = 0
            #     averageYValues = []
            #     averageXValues = []
            # reset average values

                # print(distanceToBottom,Y)

                # print circle radius
               # print('radius',i[2])
                # fl = 569.33 # old

                # fl = (p*22.5)/4.5
                # print(fl)
            # IMAGE CROPPING ON LARGEST CIRCLE FOR MORE ACCURATE DISTANCE
            # crop image to largest circle
            # UNUSED: cropping code
            # crop_added_margin = 10
            # lc = largestCircle
            # lc[0] = lc[0] - crop_added_margin
            # lc[1] = lc[1] - crop_added_margin
            # lc[2] = lc[2] + crop_added_margin
            # largestRadius_cropped = 0
            # try:
            #     crop_img_contour = countour_image[lc[1]-lc[2]:lc[1]+lc[2], lc[0]-lc[2]:lc[0]+lc[2]]
            #     crop_img_display = frame[lc[1]-lc[2]:lc[1]+lc[2], lc[0]-lc[2]:lc[0]+lc[2]]
            #     cv2.imshow('crop', crop_img_display)

            #     cropped_circles = cv2.HoughCircles(crop_img_contour,  cv2.HOUGH_GRADIENT,
            #         ValueMap('minDist', 'slider2'),
            #         ValueMap('maxDist', 'slider2'),
            #         param1=ValueMap('parem1', 'slider2'),
            #         param2=ValueMap('parem2', 'slider2'),
            #         minRadius=lc[2]-crop_added_margin*2,
            #         maxRadius=lc[2])
            #     if cropped_circles:
            #         # display cropped circles on crop_img_display
            #         cropped_circles = np.uint16(np.around(cropped_circles))
            #         largestRadius_cropped = 0
            #         for i in cropped_circles[0,:]:
            #             cv2.circle(crop_img_display,(i[0],i[1]),i[2],(0,255,0),2)
            #             cv2.circle(crop_img_display,(i[0],i[1]),2,(0,0,255),3)
            #             r = i[2]
            #             if (r > largestRadius_cropped):
            #                 largestRadius_cropped = r
            #         print('radius',i[2])
            #         l = getDistance(largestRadius_cropped)
            #         print('distance',l)
            #     else:
            #        print('crop error')
            # except Exception as e:
            #     print('no cropped circles found',[lc[1]-lc[2],lc[1]+lc[2], lc[0]-lc[2],lc[0]+lc[2]])
            # if (largestRadius_cropped != 0):
            #     getDistance(largestRadius_cropped)

    if (NOGUI == False):
        cv2.imshow('detected', masked_image)
        cv2.imshow('mask', mask)
        cv2.imshow('edges', edged)
        cv2.imshow('orginal', orginalFrame)

        cv2.imshow('countour', countour_image)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            exit()
        # key for s
        if key == ord('s'):
            print('printing values:')
            values = {}
            for i in valueTracker:
                values[i] = ValueMap(i[0], i[1])
            print(values)
            print('\n')
        if key == ord('o'):
            print("wrote output")
            ret, frame = camera.read()
            cv2.imwrite('frame.png', frame)


# loop
while True:
    # Capture frame
    if (img_path == None):  # read frame from camera
        findCircle(camera.read()[1])
    elif (img_path != None):  # read frame from file
        findCircle(cv2.imread(img_path))

camera.release()
cv2.destroyAllWindows()
