# 2022CoprocessorVision
Repository for python and other code belonging to the Pi/Jetson coprocessor

## Real Time Multi-Color Ball Distance Detection

Information regarding how the code works can be found [here](https://docs.google.com/document/d/1T7HtNdfvn1StiSUksobv4tKdpAKSKY8LUWGizXgedSI/edit?usp=sharing).

## Usage

This section details how to properly use this program.

when using in a match or setting up the PI for optimal use refer to checklist.txt

## Versions

BallDetectionDebug.py: GUI Debug program used for tuning the ball detection on a computer

streamPlusBallDetection: Ball detection and video streaming for a raspberry pi

### Installing Dependencies

For a regular user, use `pip install opencv-python==4.5.3.56` Consider using a virtual environment as this version of OpenCv is very specific.

For using on a Raspberry Pi, run:
- `sudo bash install.sh`
- `sudo bash setupPI.sh`
Now the program should start automatically when the Pi boots

### Running the Program

Run:

- `git clone https://github.com/Robotiators-888/2022CoprocessorVision.git`
- `cd 2022CoprocessorVision`
- `python3 BallDetectionDebug.py [-debug/-release/-nogui]`

### Arguments

- `-debug` runs in debug mode, with GUI slider to adjust parameters and display images
- `-release` runs without a GUI, and send data to the robot
- `-nogui` runs without a GUI in debug mode.
