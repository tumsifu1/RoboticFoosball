
"""
motor.py: runs motor driving code when required rod movements are given
"""

from config import *

from main import player_ys

from adafruit_servokit import ServoKit

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
    myKit=ServoKit(channels=16)
    import time
    while True:
        for i in range(0,90,1):
            myKit.servo[rod_move].angle=i
            time.sleep(0.01)
        for i in range(90,0,-1):
            myKit.servo[rod_move].angle=i
            time.sleep(0.01)