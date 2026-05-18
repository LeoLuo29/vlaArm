import RPi.GPIO as PGIO
import time

LED1 = 17 

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED1, GPIO.OUT)




## welcomming message
print("welcome to vlaArm mainloop")

##mainloop
while True:
    cntLoop = 0
    print("________loop" + str(cntLoop) + "_______")