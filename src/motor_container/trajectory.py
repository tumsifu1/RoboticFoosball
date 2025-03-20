"""
trajectory.py: functions to determine where the ball is projected to go
"""

import numpy as np
import threading

from config import *
from motor import trigger_motor

#Function that will calculate the vertical and horizontal velocity components
def get_velocities(ball_pos_1, ball_pos_2):
    # print("----enter get_velocities----")

    #calculate distances moved and time difference via 
    distance_x = ball_pos_2[0] - ball_pos_1[0]
    distance_y = ball_pos_2[1] - ball_pos_1[1]
    delta_t = ball_pos_2[2] - ball_pos_1[2]

    if delta_t == 0:
        return

    #calculate velocity components
    v_x = distance_x / delta_t
    v_y = distance_y / delta_t

    #call function to detect which rod needs to be moved
    get_trajectory(ball_pos_2, v_x, v_y)


def get_trajectory(ball_pos_2, v_x, v_y):
    print("---- enter get_trajectory ----")

    if v_x >= 0:
        return

    rod_positions = { 0: RIGHT_ROD_X, 1: MIDDLE_ROD_X, 2: LEFT_ROD_X }

    threads = []
    x = ball_pos_2[0]
    y = ball_pos_2[1]

    # Define which rods to trigger based on ball x
    if x > RIGHT_ROD_X - 40:
        trigger_rods = [0, 1, 2]
    elif MIDDLE_ROD_X - 40 < x <= RIGHT_ROD_X - 40:
        trigger_rods = [1, 2]
    elif LEFT_ROD_X  - 40 < x <= MIDDLE_ROD_X - 40:
        trigger_rods = [2]
    else:
        return

    for rod_move in trigger_rods:
        CURR_ROD_X = rod_positions[rod_move]
        t_intercept = (CURR_ROD_X - x) / v_x

        if t_intercept < 0:
            continue

        y_rod_new = y + v_y * t_intercept
        y_rod_new = max(TABLE_TOP + HALF_PLAYER, min(y_rod_new, TABLE_BOTTOM - HALF_PLAYER))

        t = threading.Thread(
            target=trigger_motor, 
            args=(rod_move, y_rod_new, t_intercept), 
            name=f"RodThread-{rod_move}"
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("---- exit get_trajectory ----\n")
