from cscore import CameraServer, VideoSource, UsbCamera, MjpegServer, CvSink,VideoMode
import cv2
import numpy as np


CameraServer = CameraServer()
CameraServer.enableLogging()

camera = CameraServer.startAutomaticCapture()


width = 640
height = 480

camera.setResolution(width, height)
# set video mode
# for a playstation eye camera with grayscale
camera.setVideoMode(VideoMode.PixelFormat.kGray, width, height, 30)


# get a CvSink. This will capture images from the camera
cvSink = CameraServer.getVideo()
# (optional) Setup a CvSource. This will send images back to the Dashboard
outputStream = CameraServer.putVideo("FrontIntakeCamera", width, height)



# Allocating new images is very expensive, always try to preallocate
img = np.zeros(shape=(height, width, 3), dtype=np.uint8)
# loop over images and send to dashboard
while True:
    # Tell the CvSink to grab a frame from the camera and put it
    # in the source image.  If there is an error notify the output.
    time, img = cvSink.grabFrame(img)
    if time == 0:
        # Send the output the error.
        outputStream.notifyError(cvSink.getError());
        # skip the rest of the current iteration
        continue
    # Draw a rectangle
    #cv2.rectangle(img, (0, 0), (width, height), (255, 0, 0), 5)
    # Give the output stream a new image to display
    print('sending frame')
    outputStream.putFrame(img)
    mjpegServer.putFrame(img)
