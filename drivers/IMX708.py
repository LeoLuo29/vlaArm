#!/home/leoluo52/miniconda3/envs/vlaArm313/bin/python
from picamera2 import Picamera2
picam2 = Picamera2()
picam2.start_and_capture_file("test2.jpg")