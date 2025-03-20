from config import *
from time import sleep

def game_init():

    for i in range(3):
        myKit.servo[i * 2].angle = 90
        myKit.servo[i * 2].angle = 25
        sleep(0.25)
        myKit.servo[i * 2].angle = 90
        sleep(0.5)



