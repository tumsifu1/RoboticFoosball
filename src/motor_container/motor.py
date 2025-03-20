# """
# motor.py: runs motor driving code when required rod movements are given
# """

from config import *

from time import sleep
from threading import Lock

servo_lock = Lock()

def trigger_motor(rod_move, y_rod_new, t_intercept):

    if rod_move == -1:
        return

    # Determine player on rod
    if y_rod_new < TABLE_DIV_1_Y:
        player_index = 0  # top
    elif TABLE_DIV_1_Y <= y_rod_new < TABLE_DIV_2_Y:
        player_index = 1  # middle
    else:
        player_index = 2  # bottom

    current_player_y = player_ys[rod_move][player_index]
    movement_amount = y_rod_new - current_player_y

    motor_drive(rod_move, movement_amount, t_intercept)


def motor_drive(rod_move, movement_amount, t_intercept):
    sleep_amt = abs(movement_amount) / TABLE_HEIGHT * 0.25
    # Movement phase (servo direction depends on sign of movement)
    if movement_amount > 0:
        with servo_lock:
            myKit.servo[rod_move * 2].angle = 90
            myKit.servo[rod_move * 2].angle = 160
        sleep(sleep_amt*1.5)
        with servo_lock:
            myKit.servo[rod_move * 2].angle = 90
        sleep(0.01)

    elif movement_amount < 0:
        with servo_lock:
            myKit.servo[rod_move * 2].angle = 90
            myKit.servo[rod_move * 2].angle = 25
        sleep(sleep_amt*1.5)
        with servo_lock:
            myKit.servo[rod_move * 2].angle = 90
        sleep(0.01)

    # Time until shooting
    shot_delay = min(0.5, max(0, t_intercept*0.001 - sleep_amt - 0.09))
    sleep(shot_delay)

    
        # Shoot the ball
    with servo_lock:
        myKit.servo[rod_move * 2 + 8].angle = 0
    sleep(0.01)
    with servo_lock:
        myKit.servo[rod_move * 2 + 8].angle = 90
    sleep(0.1)
    with servo_lock:
        myKit.servo[rod_move * 2 + 8].angle = 0
    sleep(0.01)

    # Update player positions
    for i in range(3):
        player_ys[rod_move][i] += movement_amount