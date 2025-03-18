
"""
motor.py: runs motor driving code when required rod movements are given
"""

from config import *

from numpy import arange

from time import sleep


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
    print(movement_amount)

    print(f"Player {player_index + 1} on rod {rod_move} moving from {current_player_y} to {y_rod_new:.2f}")

    if abs(movement_amount) > MIN_MOVEMENT:
        motor_drive(rod_move, movement_amount)


def motor_drive(rod_move, movement_amount):
    
    print(f"Moving rod {rod_move} by {movement_amount:.2f} pixels.")

    if (movement_amount > 0):
        myKit.servo[rod_move].angle = 90
        myKit.servo[rod_move].angle = 150
        sleep(0.25)
        myKit.servo[rod_move].angle = 90
        sleep(0.02)

    
    if (movement_amount < 0):
        myKit.servo[rod_move].angle = 70
        sleep(0.15)
        
        myKit.servo[rod_move].angle = 75
        time.sleep(0.5)
        myKit.servo[rod_move].angle = 90
  
    # update player positions
    for i in range(3):
        player_ys[rod_move][i] += movement_amount


    return