
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
    print(myKit.servo[0].angle)

    # Set the actuation range to something standard
    myKit.servo[0].actuation_range = 180

    # Shrink the actual PWM range (default is around 500 to 2500)
    myKit.servo[0].set_pulse_width_range(1500, 2000)  # very small physical motion range

    myKit.servo[0].angle =0

    
    # Move right from 90 to 100
    print("Moving right...")
    for i in range(90, 101, 1):
        print(f"→ Angle: {i}")
        myKit.servo[0].angle = i
        time.sleep(0.05)

    time.sleep(3)

# Move left back to 90 (including 90)
    print("Moving left...")
    for i in range(100, 89, -1):
        print(f"← Angle: {i}")
        myKit.servo[0].angle = i
        time.sleep(0.05)

    print("Done.")


    


    myKit.servo[0].angle =0

    time.sleep(5)
    
    # # # Release servo signal
    # # myKit.servo[0].angle = None

    # myKit.servo[1].angle = 90

    # # print("shot")
    
    # for i in range(90,15,-1):
    #     myKit.servo[1].angle=i
    #     time.sleep(0.003)


    # myKit.servo[1].angle = 15
    # print("shot done")

    # myKit.servo[1].angle = None


    return