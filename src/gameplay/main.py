import numpy as np

from game_init import game_init
from trajectory import *
import config


"""
main.py:    initializes game state and contains game loop that
            executes the functions when input is received
"""

from config import *

ball_old, ball_cur = game_init()

while(True):


    ball_old = (400, 500, 10000)  # x, y, timestamp
    ball_new = (410, 530, 10100)  # y increased → moving down
    get_velocities(ball_old, ball_new)



    ball_old = (410, 500, 10000)
    ball_new = (450, 470, 10100)  # y decreased → moving up
    get_velocities(ball_old, ball_new)
    
    
    # get_velocities(ball_old, ball_new)

    # print("finished")

    # time.sleep(5)



    break

    print("hi")