import numpy as np

from game_init import game_init
from trajectory import *
from config import *


"""
main.py:    initializes game state and contains game loop that
            executes the functions when input is received
"""

from config import *

count = 0

while(True):

    ball_old = (500, 500, 10000)
    ball_new = (600, 600, 10100)   # → v_x > 0, v_y > 0
    get_velocities(ball_old, ball_new)

    ball_old = (600, 500, 10000)
    ball_new = (700, 400, 10100)   # → v_x > 0, v_y < 0
    get_velocities(ball_old, ball_new)

    print(player_ys)

    
    
    # get_velocities(ball_old, ball_new)

    # print("finished")

    # time.sleep(5)

    count+=1

    if count == 5:
        break

    print("hi")