# 2022CoprocessorVision
Repository for python and other code belonging to the Pi/Jetson coprocessor

dependency: name:"opencv-python", "version": "4.5.3.56"

usage: python3 findCircle2.py [-debug|-release|-nogui]
-debug: runs in debug mode, with gui sliders to adjust parameters and displays images
-release: runs without a gui, sends data over UDP to your robot
-nogui: runs without a gui in debug mode
