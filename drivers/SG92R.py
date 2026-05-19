import RPi.GPIO as GPIO
import time

## constants
SERVO1_PIN = 18



class SG92R:
    def __init__(self,pinNum,):
        self.pinNum = pinNum
        GPIO.setup(self.pinNum, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pinNum, 50)
        self.pwm.start(0)
        self.angle = 90
        self.duty = 2.5 + (self.angle / 180) * 9.5
    
    def set_angle(self,newAngle):
        ## Move servo to angle (0–180 degrees)
        # Map 0-180 degrees to 2.5–12% duty cycle
        self.angle = newAngle
        self.duty = 2.5 + (self.angle / 180) * 9.5
        self.pwm.ChangeDutyCycle(self.duty)
        time.sleep(0.3)  # Wait for servo to reach position
        self.pwm.ChangeDutyCycle(0)  # Stop sending signal (reduces jitter)





## servo module; run in the mainloop
class Mod_SG92R:
    def __init__(self):
        self.servo1 = SG92R(pinNum = 18)
    

    def onRun(self, state = False):
        if state == False:
            return
        
        self.servo1.set_angle(90)
        time.sleep(1)
        self.servo1.set_angle(180)
        time.sleep(1)
        

