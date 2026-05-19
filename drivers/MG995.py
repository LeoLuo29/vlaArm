import RPi.GPIO as GPIO
import time



class MG995:
    def __init__(self,pinNum):
        self.pinNum = pinNum 
        GPIO.setup(self.pinNum, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pinNum, 50)
        self.pwm.start(0)
        self.angle = 90
        self.duty = 2.5 + (self.angle / 180) * 10


    def setAngle(self,newAngle):
        self.angle = newAngle
        self.duty = 2.5 + (self.angle / 180) * 10
        self.pwm.ChangeDutyCycle(self.duty)
        time.sleep(0.3)  # Wait for servo to reach position
        self.pwm.ChangeDutyCycle(0)  # Stop sending signal (reduces jitter)


    ## Module
class Mod_MG995:
    def __init__(self, pinNum):
        self.armJoint_1 = MG995(pinNum)


    def onRun(self):
        self.armJoint_1.setAngle(0)
        time.sleep(1)
        self.armJoint_1.setAngle(180)
        time.sleep(1)

    

