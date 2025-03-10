import numpy as np

from game_init import game_init
from trajectory import get_velocities
import config

from time import sleep

"""
main.py:    initializes game state and contains game loop that
            executes the functions when input is received
"""

from config import *

player_ys, ball_old, ball_cur = game_init()

while(True):
    ball_old = (380, 550, 100)
    ball_new = (390, 560, 110)

    get_velocities(ball_old, ball_new)

    sleep(5000)

    ball_old = (700, 400, 5050)
    ball_new = (705, 408, 5053)

    sleep(5000)

    get_velocities(ball_old, ball_new)

    sleep(5000)

    ball_old = (1200, 401, 10003)
    ball_new = (1208, 421, 10007)

    get_velocities(ball_old, ball_new)

    #Implement zmq accepts, zmq vals, calls get_trajectory
    #when value changes
    print("hi")