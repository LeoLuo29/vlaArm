import RPi.GPIO as GPIO
import time

# PWM: 50 Hz (T = 20 ms)
# Pulse: 0.5 ms (0°) → 2.5 ms (max_angle)
# Duty % = pulse_ms / 20 ms * 100  →  0.5ms = 2.5%,  2.5ms = 12.5%


class RDX3225:
    def __init__(self, pinNum, max_angle=270):
        self.pinNum    = pinNum
        self.max_angle = max_angle
        GPIO.setup(self.pinNum, GPIO.OUT)
        self.pwm = GPIO.PWM(self.pinNum, 50)
        self.pwm.start(0)
        self.setAngle(0)

    def _duty(self, angle):
        return 2.5 + (angle / self.max_angle) * 10.0


    def setAngle(self, angle):
        angle      = max(0, min(self.max_angle, angle))
        self.angle = angle
        self.pwm.ChangeDutyCycle(self._duty(angle))
        time.sleep(0.3)
        self.pwm.ChangeDutyCycle(0)  # stop signal to reduce jitter


    def returnMiddle(self):
        self.setAngle(180)
    

    def returnDefault(self):
        self.setAngle(0)



class Mod_RDX3225:
    def __init__(self, pinNum):
        self.servo = RDX3225(pinNum)

    def onRun(self):
        self.servo.returnMiddle()
        time.sleep(1)
        self.servo.returnDefault()
        time.sleep(1)
