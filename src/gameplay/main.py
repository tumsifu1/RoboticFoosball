import numpy as np

from game_init import game_init
from trajectory import *
import config
import time
import zmq



"""
main.py:    initializes game state and contains game loop that
            executes the functions when input is received
"""

from config import *

ball_old, ball_cur = game_init()

context = zmq.Context()
socket = context.socket(zmq.REP)

while(True):


    # ball_old = (382, 551, 12003)
    # ball_new = (395, 558, 12048)
    
    # print(ball_old)
    # print(ball_new)

    # get_velocities(ball_old, ball_new)

    # print("finished")

    # time.sleep(5)



    break

    print("hi")