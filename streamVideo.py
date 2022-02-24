from cscore import CameraServer, VideoSource, UsbCamera, MjpegServer, CvSink,VideoMode


import cv2
import numpy as np


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

camera.setResolution(width, height)
# set video mode
# for a playstation eye camera 
camera.setVideoMode(VideoMode.PixelFormat.kYUYV, width, height, 30)


# get a CvSink. This will capture images from the camera
cvSink = CameraServer.getVideo()
# (optional) Setup a CvSource. This will send images back to the Dashboard
outputStream = CameraServer.putVideo("FrontIntakeCamera", width, height)

print('sending video to http://'+ourIp+':1181/stream.mjpg')


# Allocating new images is very expensive, always try to preallocate
img = np.zeros(shape=(480, 640, 3), dtype=np.uint8)

while True:
    # Tell the CvSink to grab a frame from the camera and put it
    # in the source image.  If there is an error notify the output.
    time, img = cvSink.grabFrame(img)
    if time == 0:
        # Send the output the error.
        outputStream.notifyError(cvSink.getError())
        # skip the rest of the current iteration
        continue

    # Put a rectangle on the image
    #cv2.rectangle(img, (100, 100), (400, 400), (255, 255, 255), 5)
    halfImg = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)


    # Give the output stream a new image to display
    outputStream.putFrame(halfImg)
