#!/usr/bin/env python3

from distutils.log import error
import imp
from pickle import TRUE
import cv2
from cv2 import erode
import numpy as np
import time
import json
import math
import sys

import math
import struct
import socket
from threading import Thread, Event
import threading

global sock


# is in debug mode
DEBUG = False
# are we using gui windows
NOGUI = False

BallColor = "Red"

# path to input image
img_path = None

#TODO:
# 1 - allow specifying camera id and red/blue ball in args
# 2 send UDP true/false for red/blue ball

# get command line arguments
if len(sys.argv) > 1:
    #print(len(sys.argv))
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
        print('r: start recording video')
        print('c: toggle visualization of Hcircles')
        print('t: start auto-tuning of input values')
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

#SECTION: networktables
if (DEBUG==False):
    from networktables import NetworkTables
    NetworkTables.setIPAddress("10.8.88.86")
    NetworkTables.setClientMode()
    NetworkTables.initialize()
    nt = NetworkTables.getTable("BallDetection")

# sends udp data to roboRio ip
# format: float x, float y, boolean ballDetected
def sendNT(x, y, detected):
    nt.putNumber('x', x)
    nt.putNumber('y', y)
    nt.putBoolean('ballDetected', detected)

    return "Success"

# SECTION: math functions
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

def nothing(x):
    pass


def inchesToMeters(inches):
    return inches * 0.0254

valueTracker = []

# sets up GUI sliders for debug mode
# SECTION: GUI sliders
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
    cv2.createTrackbar('CONSTANT', 'slider2', -1, 400, nothing)
    valueTracker.append(('CONSTANT', 'slider2'))


OrangeBallValues = {('hue', 'slider'): 0, ('sat', 'slider'): 81, ('val', 'slider'): 65, ('hue2', 'slider2'): 12, ('sat2', 'slider2'): 255, ('val2', 'slider2'): 255,
                    ('th1', 'slider2'): 119, ('th2', 'slider2'): 53, ('minRadius', 'slider2'): 52, ('maxRadius', 'slider2'): 100, ('parem1', 'slider2'): 15, ('parem2', 'slider2'): 26}
#RedTrainingValues = {('hue', 'slider'): 0, ('sat', 'slider'): 71, ('val', 'slider'): 13, ('hue2', 'slider2'): 16, ('sat2', 'slider2'): 255, ('val2', 'slider2'): 255, ('th1', 'slider2'): 184, ('th2', 'slider2')
#                      : 53, ('minRadius', 'slider2'): 30, ('maxRadius', 'slider2'): 200, ('parem1', 'slider2'): 19, ('parem2', 'slider2'): 21, ('minDist', 'slider2'): 1, ('maxDist', 'slider2'): 100, ('HConstant', 'slider2'): 2000}
BlueTrainingValues = {('hue', 'slider'): 32, ('sat', 'slider'): 71, ('val', 'slider'): 13, ('hue2', 'slider2'): 146, ('sat2', 'slider2'): 255, ('val2', 'slider2'): 255, ('th1', 'slider2'): 184, ('th2', 'slider2')
                       : 53, ('minRadius', 'slider2'): 24, ('maxRadius', 'slider2'): 150, ('parem1', 'slider2'): 19, ('parem2', 'slider2'): 25, ('minDist', 'slider2'): 1, ('maxDist', 'slider2'): 100, ('HConstant', 'slider2'): 2000}
#RedTrainingValues = {('hue', 'slider'): 0, ('sat', 'slider'): 52, ('val', 'slider'): 13, ('hue2', 'slider2'): 16, ('sat2', 'slider2'): 255, ('val2', 'slider2'): 255, ('th1', 'slider2'): 184, ('th2', 'slider2'): 53, ('minRadius', 'slider2'): 30, ('maxRadius', 'slider2'): 150, ('parem1', 'slider2'): 19, ('parem2', 'slider2'): 21, ('minDist', 'slider2'): 1, ('maxDist', 'slider2'): 100, ('HConstant', 'slider2'): 2000}

# The input values for finding the Red Ball
#RedTrainingValues ={('hue', 'slider'): 0, ('sat', 'slider'): 52, ('val', 'slider'): 13, ('hue2', 'slider2'): 16, ('sat2', 'slider2'): 255, ('val2', 'slider2'): 255, ('th1', 'slider2'): 184, ('th2', 'slider2'): 53, ('minRadius', 'slider2'): 30, ('maxRadius', 'slider2'): 150, ('parem1', 'slider2'): 19, ('parem2', 'slider2'): 28, ('minDist', 'slider2'): 1, ('maxDist', 'slider2'): 100, ('HConstant', 'slider2'): 2000,('CONSTANT', 'slider2'):2500}#
#RedTrainingValues = {('hue', 'slider'): 0, ('sat', 'slider'): 52, ('val', 'slider'): 13, ('hue2', 'slider2'): 16, ('sat2', 'slider2'): 255, ('val2', 'slider2'): 255, ('th1', 'slider2'): 184, ('th2', 'slider2'): 53, ('minRadius', 'slider2'): 21, ('maxRadius', 'slider2'): 102, ('parem1', 'slider2'): 19, ('parem2', 'slider2'): 14, ('minDist', 'slider2'): 1, ('maxDist', 'slider2'): 100, ('HConstant', 'slider2'): 2000, ('CONSTANT', 'slider2'): 400}
#RedTrainingValues = {('hue', 'slider'): 0, ('sat', 'slider'): 52, ('val', 'slider'): 13, ('hue2', 'slider2'): 16, ('sat2', 'slider2'): 255, ('val2', 'slider2'): 255, ('th1', 'slider2'): 184, ('th2', 'slider2'): 53, ('minRadius', 'slider2'): 21, ('maxRadius', 'slider2'): 102, ('parem1', 'slider2'): 19, ('parem2', 'slider2'): 17, ('minDist', 'slider2'): 1, ('maxDist', 'slider2'): 100, ('HConstant', 'slider2'): 2000, ('CONSTANT', 'slider2'): 400}
RedTrainingValues = {('hue', 'slider'): 0, ('sat', 'slider'): 52, ('val', 'slider'): 13, ('hue2', 'slider2'): 16, ('sat2', 'slider2'): 255, ('val2', 'slider2'): 255, ('th1', 'slider2'): 184, ('th2', 'slider2'): 53, ('minRadius', 'slider2'): 21, ('maxRadius', 'slider2'): 102, ('parem1', 'slider2'): 19, ('parem2', 'slider2'): 25, ('minDist', 'slider2'): 1, ('maxDist', 'slider2'): 100, ('HConstant', 'slider2'): 2000, ('CONSTANT', 'slider2'): 400}
# current ^

# SECTION: Average Dist Que
# Que holding last n Y distances
averageYValues = [] 
# Que holding last n X distances
averageXValues = []
# max amount of items in the que
averageXQueueLength = 4
averageYQueueLength = 4

# The current frame number
global averageCounter
averageCounter = 0


ValueMap = {}
global Visualize 
# to show out from HCircles or not, on frame
Visualize = True

# gets distance x and y in inches to ball from pixelRadius of Ball. 
# And X pixels from the center of the image to center of ball
#SECTION: Get Distance 
#NOTE: Refere to documentation for more info
#https://docs.google.com/document/d/1T7HtNdfvn1StiSUksobv4tKdpAKSKY8LUWGizXgedSI/edit
def getDistance(pr,pxFromCenterX):
    fl = 539.0
    r = 4.5
    d = ((fl*r)/pr)
    # use Hconstant

    #CONSTANT = ValueMap[('CONSTANT', 'slider2')]
    #c = CONSTANT/pr
    try:
        #fov = 56
        #fov in radians
        Rfov = 0.97738438111682

        xt = math.tan((Rfov/2))*d


        n = 640
        p = pxFromCenterX

        xdist = (2*xt*p)/n
        ydist = d
        return [xdist, ydist]
    except ValueError:
        xdist = 0
        ydist = 0

#SECTION: ROBOT VALUES 
#RedTrainingValues = {('hue', 'slider'): 0, ('sat', 'slider'): 74, ('val', 'slider'): 49, ('hue2', 'slider2'): 12, ('sat2', 'slider2'): 255, ('val2', 'slider2'): 255, ('th1', 'slider2'): 112, ('th2', 'slider2'): 53, ('minRadius', 'slider2'): 0, ('maxRadius', 'slider2'): 150, ('parem1', 'slider2'): 19, ('parem2', 'slider2'): 27, ('minDist', 'slider2'): 1, ('maxDist', 'slider2'): 100}
#RedTrainingValues = {('hue', 'slider'): 0, ('sat', 'slider'): 156, ('val', 'slider'): 73, ('hue2', 'slider2'): 179, ('sat2', 'slider2'): 255, ('val2', 'slider2'): 255, ('th1', 'slider2'): 112, ('th2', 'slider2'): 53, ('minRadius', 'slider2'): 30, ('maxRadius', 'slider2'): 150, ('parem1', 'slider2'): 19, ('parem2', 'slider2'): 14, ('minDist', 'slider2'): 1, ('maxDist', 'slider2'): 100, ('HConstant', 'slider2'): 2000}

# put input values in a dictionary called ValueMap
# ValueMap is used to store values functions use
# #example: ValueMap[('hue', 'slider')] = 50
# It stores a hue that can represent the color of the ball a function
# 'slider' shows which slider this maps to for GUI sliders
if (BallColor == 'Red'):
    for i in RedTrainingValues:
        ValueMap[(i[0], i[1])] = RedTrainingValues[i]
        if (NOGUI == False):
            cv2.setTrackbarPos(i[0], i[1], RedTrainingValues[i])
elif (BallColor == 'Blue'):
    for i in BlueTrainingValues:
        ValueMap[(i[0], i[1])] = BlueTrainingValues[i]
        if (NOGUI == False):
            cv2.setTrackbarPos(i[0], i[1], BlueTrainingValues[i])

# id of camera, usally 0
# cameraId = 4

# # create a camera object
# camera = cv2.VideoCapture(cameraId)


# SECTION: findCircle main method for finding the ball

class FindBallThread:
    foundBall = None

def findBall():
    print("finding balls!")
    global Visualize
    frame = currentFrame.frame.copy()
    #frame = cv2.imread(img_path)  
    orginalFrame = frame.copy()


   # use Umat for 3 fps boost
    umatFrame = cv2.UMat(frame)

     # change image color space to HSV
    hsv = cv2.cvtColor(umatFrame, cv2.COLOR_BGR2HSV)

    # SECTION: Mask image to only show color of ball
    lower_color = np.array([ValueMap[('hue', 'slider')], ValueMap[(
        'sat', 'slider')], ValueMap[('val', 'slider')]])
    upper_color = np.array([ValueMap[('hue2', 'slider2')], ValueMap[(
        'sat2', 'slider2')], ValueMap[('val2', 'slider2')]])

    # mask the image: only keep colors in between lower_color and upper_color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # SECTION: Filter out noise/
    # remove small objects
    
    startingTime = time.time()
    mask = cv2.erode(mask, None, iterations=4)
    mask = cv2.dilate(mask, None, iterations=4)
    endingTime = time.time()
    deltaTime = endingTime - startingTime
    #print("Erode and Dilate Time: " + str(deltaTime))

    # apply the mask to the image
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)
    
    # gray version of masked image
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    #  shadowRectangle = empty frame
    shadowRectangle = np.zeros(frame.shape, np.uint8)


    # detect average location of brightest pixels in grey
    # and use that as the center of the circle
    # DEBUGING method!
    startingTime = time.time()
    endingTime = time.time()
    deltaTime = endingTime - startingTime
    #print("Average Location Time: " + str(deltaTime))

    # do a GaussianBlur to remove noise
    startingTime = time.time()
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    endingTime = time.time()
    deltaTime = endingTime - startingTime

    # get percentage of white pixels in mask
    # SECTION: detect laggy frame and skip
    startingTime = time.time()
    whitePixels = np.count_nonzero(gray)
    if (whitePixels > 60966):
        print('frame will cause extreme lag')
        if (DEBUG == False):
            sendNT(0, 0, False)
        return
    else:
        pass
    endingTime = time.time()
    deltaTime = endingTime - startingTime
    
    #SECTION: HCircles
    # half the image going to HCircles to reduce lag
    halfGray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    # turn Umat halfGray into a normal image
    halfGray = cv2.UMat.get(halfGray)

    startingTime = time.time()

    circles = cv2.HoughCircles(halfGray,  cv2.HOUGH_GRADIENT,
                                ValueMap[('minDist', 'slider2')],
                                ValueMap[('maxDist', 'slider2')],
                                param1=ValueMap[('parem1', 'slider2')],
                                param2=ValueMap[('parem2', 'slider2')],
                                minRadius=ValueMap[(
                                    'minRadius', 'slider2')],
                                maxRadius=ValueMap[('maxRadius', 'slider2')])
    endingTime = time.time()
    deltaTime = endingTime - startingTime
    
    # SECTION: found circles
    if circles is not None:
        # varibles hold one closest circle
        largestDis = 0
        largestRadius = 0
        closestCircle = None  

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # multiply all circles by 2 to get back to original size
            i[0] = i[0] * 2
            i[1] = i[1] * 2
            i[2] = i[2] * 2
            if (Visualize): # draw extra information on screen if v toggle pressed
                cv2.circle(frame, (i[0], i[1]), i[2], (162, 255, 0), 2)
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
                # draw on maske_image
                cv2.circle(masked_image, (i[0], i[1]), i[2], (162, 255, 0), 2)
                cv2.circle(masked_image, (i[0], i[1]), 2, (0, 0, 255), 3)
            r = i[2]
            # check if radius is on left or right of screen
            # find ditstance from center of circle to bottom of screen
            distanceToBottom = frame.shape[0] - i[1] - i[2]
            # #print('distanceToBottom',distanceToBottom)
            A = 163.5
            B = -879.9
            C = 1020.0

            try:
                dis = None
                pxFromCenterX = i[0] - frame.shape[1] / 2

                print('before dis')
                dis = getDistance(r,pxFromCenterX)
                print('after dis')

                # TODO: SHADOW DETECTION
                center = [i[0], i[1]]
                center[1] = center[1] + int(i[2]*1.2)
                # plot center on frame
                cv2.circle(frame, center, 2, (255, 0, 0), 3)
                

                # SECTION: shadow detection
                # create a rectangle under the ball proportional to its radius
                # average the HSV Value in the rectangle as averageColor
                shadowRectangle = frame[center[1] - int(i[2]/7):center[1] + int(
                    i[2]/7), center[0] - int(i[2]/2):center[0] + int(i[2]/2)]
                # averag3e the rectangle
                averageColor = 0
                startingTime = time.time()
                for x in range(shadowRectangle.shape[0]):
                    for y in range(shadowRectangle.shape[1]):
                        averageColor += shadowRectangle[x][y][2]
                averageColor = averageColor / \
                    (shadowRectangle.shape[0]*shadowRectangle.shape[1])
                endingTime = time.time()
                deltaTime = endingTime - startingTime
               

                print('averageCOlor',averageColor)
                # Value is how light or dark the color is
                # if averageColor in this range its color of balls shadow
                if (averageColor > 15 and averageColor < 80):
                    pass
                else:
                    # skip frame if not dark
                    break


                # SECTION: ground detection
                # Using the Y distance to the ball and a Graph find the pixels from bottom of ball to bottom of screen
                X = dis[1]
                A = 283.8
                B = -163800.0
                C = 24070000.0
                Y = Caunchy(X, A, B, C)

                #started at 21 going up by one inch per
                #went up to 64 inches

                #distanceToBottom pixels from bottom of ball to bottom of screen
                #Y predicted pixel distance from ball to bottom of screen
                # get the ratio between them
                onGroundRatio = distanceToBottom/Y
                # if there close then the ball must be on the ground
                #if (onGroundRatio > -0.2 and onGroundRatio < 1.5):
                if (onGroundRatio > 0.6 and onGroundRatio < 2.2):
                    totalDis = abs(dis[0]) + dis[1]
                    # to find the closest ball
                    # if our radius is greater than largestRadius do this:
                    if (r > largestRadius):
                        largestDis = totalDis
                        largestRadius = r
                        closestCircle = i

                else:
                    print('notonGround', onGroundRatio)

                
            except Exception as e:
                print("error finding ball", e)
        if (closestCircle is not None): # code that runs on the closest found ball
            print('found closest circle')
            i = closestCircle # closest circle
            r = i[2] # radius
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 0, 255), 7)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            # draw on maske_image
            cv2.putText( # label the ball on frame above the ball
                frame, 'Real Ball', (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.circle(masked_image, (i[0], i[1]), i[2], (0, 0, 255), 7)
            cv2.circle(masked_image, (i[0], i[1]), 2, (0, 0, 0), 3)


            dis = None
            pxFromCenterX = i[0] - frame.shape[1] / 2
            print('pxFromCenterX', pxFromCenterX)
            print('before dis closest circle')
            dis = getDistance(r,pxFromCenterX)
            print('before dis closest circle')

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
            # get average of positions
            averageY = sum(averageYValues)/len(averageYValues)
            averageX = sum(averageXValues)/len(averageXValues)
            #print('averageY', averageY, 'averageX', averageX)

            # TODO: tune output like this to desired position

            # send data over UDP to RoboRio
            #
            if (DEBUG == False):
                sendNT(averageX, averageY, True)

            # print('averageY',averageY,'averageX',averageX)
            # draw text in the corner of the screen with averageDist
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

            FindBallThread.foundBall = True

            print('X',averageX,'Y',averageY)
            if (DEBUG == False):
                sendNT(averageY, averageY, True)
        else:
            if (DEBUG == False):
                sendNT(0, 0, False)
            print('no ball found\n')
            FindBallThread.foundBall = False
            return (0,0,False)
    else:
        # No ball found
        FindBallThread.foundBall = False
    # SECTION: bring up windows showing frames
    if (NOGUI == False):
        cv2.imshow('detected', masked_image)
        cv2.imshow('mask', mask)
        #cv2.imshow('edges', edged)
        cv2.imshow('orginal', orginalFrame)

        #cv2.imshow('countour', countour_image)
        cv2.imshow('frame', frame)
        cv2.imshow('gray', gray)

        key = cv2.waitKey(1)
        # SECTION Key Presses: 
        # check for key presses, refere to -help for what key presses do
        if key == 27 or key == ord('q'): # Q key
            exit()
        # key for s
        if key == ord('s'): # save 
            print('printing values:')
            values = {}
            for i in valueTracker:
                # get the value from the slider put in values
                values[i] = cv2.getTrackbarPos(i[0], i[1])
            print(values)
            print('\n')
        if key == ord('o'): # O key
            print("wrote output")
        if key == ord('r'): # 
            print("wrote video")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
            while(True):
                ret, frame = camera.read()
                if ret == True:
                    out.write(frame)
                else:
                    break
            out.release()
            cv2.destroyAllWindows()
        if key == ord('v'):
            Visualize = not Visualize
        
        # return values
        if FindBallThread.foundBall:
            print('x',averageX,'y',averageY)
            threading.exit()
            return (averageX, averageY, True)
        else:
            if (DEBUG == False):
                sendNT(0, 0, False)
            print('no ball found\n')
            return (0,0,False)
            


    
from cscore import CameraServer, VideoSource, UsbCamera, MjpegServer, CvSink,VideoMode


import cv2
import numpy as np
import time


from networktables import NetworkTables
NetworkTables.initialize(server='10.8.88.2')

import socket
def findIp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return(s.getsockname()[0])
    s.close()

ourIp = findIp()

CameraServer = CameraServer().getInstance()
CameraServer.enableLogging()

camera = CameraServer.startAutomaticCapture()


width = 640
height = 480

halfWidth = 320
halfHeight = 240

camera.setResolution(320, 240)
# set video mode
# for a playstation eye camera 
camera.setVideoMode(VideoMode.PixelFormat.kYUYV, width, height, 30)


# get a CvSink. This will capture images from the camera
cvSink = CameraServer.getVideo()
# (optional) Setup a CvSource. This will send images back to the Dashboard
outputStream = CameraServer.putVideo("FrontIntakeCamera", 320, 240)
# create a mjpeg server
print('sending video to http://'+ourIp+':1181/stream.mjpg')


# Allocating new images is very expensive, always try to preallocate
img = np.zeros(shape=(480, 640, 3), dtype=np.uint8)

# write a function that takes around one second to compute a number
# and return it
timeStamp = time.time()
HCStartTime = 0
class currentFrame:
    frame = None

ballDetectThread = Thread(target=findBall)

# t, img = cvSink.grabFrame(img)
# currentFrame.frame = img
# try:
#     Thread(target=ballDetectThread).start()
# except error:
#     print('findBall died: ',error)


while True:
    #Tell the CvSink to grab a frame from the camera and put it
    #in the source image.  If there is an error notify the output.
    t, img = cvSink.grabFrame(img)
    currentFrame.frame = img
    if t == 0:
    #     # Send the output the error.
        outputStream.notifyError(cvSink.getError())
    #     # skip the rest of the current iteration
        continue

    # run the ballDetect function async
    # if a ballDetect thread is already running, don't start another one
    HCRuntime = 6
    #time hough circles has been running for

    if HCStartTime+HCRuntime<time.time() or FindBallThread.foundBall != None:
        print("HCircles too old killing or ran")
        HCStartTime = time.time()
        # if thread is alive kill it
        try:
            if ballDetectThread.is_alive():
                #kill the thread
                print('is alive',ballDetectThread.is_alive())
                ballDetectThread.exit()
            print('starting new thread')
            ballDetectThread = Thread(target=findBall)
            FindBallThread.foundBall = None
            target=ballDetectThread.start()
        except error:
            print(error)
        



#         if (NOGUI == False): # get slider values then put into ValueMap
#             Newvalues = {}
#             for i in valueTracker:
#                 # get the value from the slider put in values
#                 Newvalues[i] = cv2.getTrackbarPos(i[0], i[1])
#             if (Newvalues != ValueMap):
#                 ValueMap = Newvalues


#     #SECTION: SENDING VIDEO STREAM
    halfImg = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # #print the resolution of halfImg
    # print(halfImg.shape)
    # # draw a line on halfImg
    #cv2.line(halfImg, (20, 30), (160, 180), (0, 255, 0), 40)
    # # # Give the output stream a new image to display
    # # write the image to a file
    
    outputStream.putFrame(halfImg)

#     #fps
#     # print number houghCirclesThread running

#     print(1/(time.time()-timeStamp))
#     timeStamp = time.time()



    # Tell the CvSink to grab a frame from the camera and put it
    # in the source image.  If there is an error notify the output.
    # t, img = cvSink.grabFrame(img)
    # if t == 0:
    # #     # Send the output the error.
    #     outputStream.notifyError(cvSink.getError())
    # #     # skip the rest of the current iteration
    #     continue
    # Capture frame
    # output = ()
    # toPrint = ''
    # if (img_path == None):  # read frame from camera
    #     findCircle(img)
    # elif (img_path != None):  # read frame from file
    #     findCircle(cv2.imread(img_path))

    # find fps
    # newTime = time.time()
    # deltaTime = newTime - currentTime
    # fps = 1/deltaTime
    # currentTime = newTime
    # maxFps = 5
    # maxTime = maxFps/60 # in secs
    # if (deltaTime<maxTime):
    #     waitTime = maxTime-deltaTime
    #     time.sleep(waitTime)
    # print('fps', fps)
    #print('''\n\n\n''')


# camera.release()
# cv2.destroyAllWindows()

# def helloWorld():
#     print('hello world!')

# helloWorldThread = Thread(target=helloWorld)
# findBallThread = Thread(target=findBall)
# findBallThread.start()

# helloWorldThread.start()
