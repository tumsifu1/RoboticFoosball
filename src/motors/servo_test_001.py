print("t1")
from adafruit_servokit import ServoKit
kit=ServoKit(channels=16)
print("t2")
import time
print("t3")
while True:
    for i in range(0,180,1):
        print("t4")
        kit.servo[0].angle=i
        time.sleep(0.001)
    for i in range(180,0,-1):
        print("t5")
        kit.servo[0].angle=i
        time.sleep(0.001)
