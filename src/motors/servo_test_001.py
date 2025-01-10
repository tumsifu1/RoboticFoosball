
from adafruit_servokit import ServoKit
myKit=ServoKit(channels=16)
import time
while True:
    for i in range(0,180,1):
        myKit.servo[0].angle=i
        time.sleep(0.01)
    for i in range(180,0,-1):
        myKit.servo[0].angle=i
        time.sleep(0,01)
