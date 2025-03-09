import numpy as np

from game_init import game_init

"""
main.py:    initializes game state and contains game loop that
            executes the functions when input is received
"""

from config import *

player_ys, ball_old, ball_cur = game_init()

while(True):
    
    #Implement zmq accepts, zmq vals, calls get_trajectory
    #when value changes
    print("hi")