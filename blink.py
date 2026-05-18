import time
import RPi.GPIO as GPIO



##constants
LED1 = 17

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED1, GPIO.OUT)



##mainloop
while True:
    GPIO.output(LED1, True)
    time.sleep(1)
    GPIO.output(LED1, False)
    time.sleep(1)
    print("toggled")



