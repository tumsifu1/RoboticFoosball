"""
trajectory.py: functions to determine where the ball is projected to go
"""

import numpy as np
import pandas as pd
from config import *

from motor import trigger_motor

#Function that will calculate the vertical and horizontal velocity components
def get_velocities(ball_pos_1, ball_pos_2):
    print("----enter get_velocities----")

    #calculate distances moved and time difference via 
    distance_x = ball_pos_2[0] - ball_pos_1[0]
    distance_y = ball_pos_2[1] - ball_pos_1[0]
    delta_t = ball_pos_2[2] - ball_pos_1[2]

    print("Distances + delta t: ", distance_x, " ", distance_y, " ", delta_t)

    if delta_t == 0:
        return

    #calculate velocity components
    v_x = distance_x / delta_t
    v_y = distance_y / delta_t

    print("vx and vy: ", v_x, " ", v_y)


    #call function to detect which rod needs to be moved
    get_trajectory(ball_pos_2, v_x, v_y)

#Function that determines what rod will need to have its players moved, and the trajectory of the ball
def get_trajectory(ball_pos_2, v_x, v_y):

    print("----enter get_trajectory----")
    rod_move = -1
    CURR_ROD_X = 0

    if v_x > 0:
        if ball_pos_2[0] < LEFT_ROD_X:
            rod_move = 0
            CURR_ROD_X = LEFT_ROD_X
        elif LEFT_ROD_X <= ball_pos_2[0] < MIDDLE_ROD_X:
            rod_move = 1
            CURR_ROD_X = MIDDLE_ROD_X
        elif MIDDLE_ROD_X <= ball_pos_2[0] < RIGHT_ROD_X:
            rod_move = 2
            CURR_ROD_X = RIGHT_ROD_X
        else:
            rod_move = -1

    elif v_x < 0:
        if ball_pos_2[0] > RIGHT_ROD_X:
            rod_move = 0
            CURR_ROD_X = RIGHT_ROD_X
        elif RIGHT_ROD_X >= ball_pos_2[0] > MIDDLE_ROD_X:
            rod_move = 1
            CURR_ROD_X = MIDDLE_ROD_X
        elif MIDDLE_ROD_X >= ball_pos_2[0] > LEFT_ROD_X:
            rod_move = 2
            CURR_ROD_X = LEFT_ROD_X
        else:
            rod_move = -1

    if rod_move == -1:
        return

    print("rod_move: ", rod_move)

    if abs(v_x) > 1e-6:
        t_intercept = (CURR_ROD_X - ball_pos_2[0]) / v_x
        y_rod_new = ball_pos_2[1] + v_y * t_intercept

        print("t_intercept and y_rod_new", t_intercept, y_rod_new)

    trigger_motor(rod_move, y_rod_new)