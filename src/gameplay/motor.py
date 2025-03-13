
"""
motor.py: runs motor driving code when required rod movements are given
"""

from config import *

from numpy import arange

import time



#Add case when player centre cannot move to a position (player will run into wall), move as far as possible
def trigger_motor(rod_move, y_rod_new):
    if rod_move == -1:
        return

    if y_rod_new < TABLE_DIV_1_Y:
        player_index = 0  #top player
    elif y_rod_new < TABLE_DIV_2_Y:
        player_index = 1  #middle player
    else:
        player_index = 2  #bottom player

    #curr player pos
    current_player_y = player_ys[rod_move][player_index]

    #wall proximity constraints
    if y_rod_new - HALF_PLAYER < TABLE_TOP:
        y_rod_new = TABLE_TOP + HALF_PLAYER     #Prevent the motors from over-running when the ball is projected to go right by the wall
    elif y_rod_new + HALF_PLAYER > TABLE_BOTTOM:
        y_rod_new = TABLE_BOTTOM - HALF_PLAYER

    movement_amount = y_rod_new - current_player_y

    print(f"Player {player_index + 1} on rod {rod_move} moving from {current_player_y} to {y_rod_new:.2f}")

    if abs(movement_amount) > MIN_MOVEMENT:
        motor_drive(rod_move, movement_amount)


def motor_drive(rod_move, movement_amount):
    print(f"Moving rod {rod_move} by {movement_amount:.2f} pixels.")
    #Here the code will be to convert pixels into rotations of motor, etc
        #will also implement the array changing within this
        #another place where the state array will change is within the trajectory mapping function
        #as it will have the new ball locations, only question is how often will it be run?
        #every 2 frames, every few?
    myKit.servo[rod_move].angle = None

    print("moving motor up")
    for i in arange(0, 1, 0.1):
        myKit.servo[rod_move].angle = i
        time.sleep(0.005)
    myKit.servo[rod_move].angle = None
    time.sleep(1)
    print("moving motor down")
    for i in arange(1, 0, -0.1):
        myKit.servo[rod_move].angle = i
        time.sleep(0.005)

    # Stop motor
    myKit.servo[rod_move].angle = None
    print("motor stopped")



    # counter = 0
    # while True:
    #     myKit.servo[rod_move].angle = None

    #     print("moving motor")
    #     for i in arange(1,0,-0.1):
    #         myKit.servo[rod_move].angle=i
    #         time.sleep(10)
    #     print("stopping motor")

    #     print("moving motor")
    #     for i in arange(0,1,0.1):
    #         myKit.servo[rod_move].angle=i
    #         time.sleep(10)
    #     myKit.servo[rod].angle = None
    #     print("stopping motor")

    #     # print("shot")
    #     # for i in range(0,45,1):
    #     #     myKit.servo[1].angle=i
    #     #     time.sleep(0.01)
    #     # print("shot done")

    #     # myKit.servo[1].angle = None


    #     break

    return