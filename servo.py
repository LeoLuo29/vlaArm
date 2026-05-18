
## constants
SERVO1_PIN = 18



GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)



class Servo:
    def __init__(self,pinNum,):
        self.pinNum = pinNum
        GPIO.setup(pinNUm, GPIO.OUT)
        self.pwm = GPIO.PWM(SERVO_PIN, 50)
        self.pwm.start(0)
        self.angle = 90
        self.duty = 2.5 + (self.angle / 180) * 9.5
    
    def set_angle(self,angle):
        ## Move servo to angle (0–180 degrees)
        # Map 0-180 degrees to 2.5–12% duty cycle
        self.duty = 2.5 + (angle / 180) * 9.5
        self.pwm.ChangeDutyCycle(self.duty)
        time.sleep(0.3)  # Wait for servo to reach position
        self.pwm.ChangeDutyCycle(0)  # Stop sending signal (reduces jitter)





    ## servo module; run in the mainloop