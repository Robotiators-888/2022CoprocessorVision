# 2022CoprocessorVision
Repository for python and other code belonging to the Pi/Jetson coprocessor

## Real Time Multi-Color Ball Distance Detection

Information regarding how the code works can be found [here](https://docs.google.com/document/d/1T7HtNdfvn1StiSUksobv4tKdpAKSKY8LUWGizXgedSI/edit?usp=sharing).

## Usage

This section details how to properly use this program.

### Installing Dependencies

For a regular user, use `pip install opencv-python==4.5.3.56` Consider using a virtual environment as this version of OpenCv is very specific.

For using on a Raspberry Pi, run:
- `sudo apt-get install python3-opencv`
- `pip3 install opencv-python`


### Running the Program

Run:

- `git clone https://github.com/Robotiators-888/2022CoprocessorVision.git`
- `cd 2022CoprocessorVision`
- `python3 findCircle.py [-debug/-release/-nogui]`

### Arguments

- `-debug` runs in debug mode, with GUI slider to adjust parameters and display images
- `-release` runs without a GUI, and send data over UDP to the robot
- `-nogui` runs without a GUI in debug mode.
