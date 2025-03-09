import numpy as np
import pandas as pd

#Dimensions of elements of the tables, pixel-wise
FRAME_WIDTH = 2304
FRAME_HEIGHT = 1296

#PLAYER DIMS:
PLAYER_WIDTH = 65
WASHER_THICKNESS = 12           #Bounds will be needed to be added for inferencing or an adjustment to just push the y value of the ball within bounds
PLAYER_GAP = 252
HALF_PLAYER = 65 // 2

#INITIALLY MEASURED HOWEVER WILL NEED TO BE CENTERED EVEN BETTER
TABLE_LEFT = 310      #294 + 16       (32 diff between measurements)
TABLE_RIGHT = FRAME_WIDTH - TABLE_LEFT     #326 - 16
TABLE_TOP = 228 + WASHER_THICKNESS         #216 + 12
TABLE_BOTTOM = FRAME_HEIGHT - TABLE_TOP      #240 - 12
TABLE_HEIGHT = TABLE_BOTTOM - TABLE_TOP
TABLE_WIDTH = TABLE_RIGHT - TABLE_LEFT

#Divisions in between thirds of the table. Player will be chosen depending on which zone the ball will go to
TABLE_DIV_1_Y = TABLE_TOP + TABLE_HEIGHT // 3
TABLE_DIV_2_Y = TABLE_TOP + 2 * TABLE_HEIGHT // 3

#ROD LOCATIONS (+16)
LEFT_ROD_X = 472
MIDDLE_ROD_X = 984
RIGHT_ROD_X = 1496

#Minimum movement to prevent trigger motor to be called if it is too small (prevent unnecessary start/stops)
MIN_MOVEMENT = 10

def generate_game_state(frame_width=1280, frame_height=720):
    # -1 OOB area
    game_state = np.full((frame_height, frame_width), -1, dtype=int)
    
    #table surface is 0s
    game_state[TABLE_TOP:TABLE_BOTTOM, TABLE_LEFT:TABLE_RIGHT] = 0
    
    #Set rod positions to 1s, players are 2s
    rod_positions = [LEFT_ROD_X, MIDDLE_ROD_X, RIGHT_ROD_X]
    player_start_y = TABLE_TOP + HALF_PLAYER  # 1st player cntr

    initial_player_y = [[0,0,0],[0,0,0],[0,0,0]]
    
    for j, rod_x in enumerate(rod_positions):
        if TABLE_LEFT <= rod_x < TABLE_RIGHT:
            game_state[TABLE_TOP:TABLE_BOTTOM, rod_x] = 1
            #3 players
            for i in range(3):
                player_y = player_start_y + i * PLAYER_GAP
                initial_player_y[j][i] = player_y
                game_state[player_y, rod_x] = 2  #Centre of players is 2s. No need for width
    
    return game_state, initial_player_y

#Function that will calculate the vertical and horizontal velocity components
def get_velocities(ball_pos_1, ball_pos_2):

    #calculate distances moved and time difference via 
    distance_x = ball_pos_2[0] - ball_pos_1[0]
    distance_y = ball_pos_2[1] - ball_pos_1[0]
    delta_t = ball_pos_2[2] - ball_pos_1[2]

    if delta_t == 0:
        return

    #calculate velocity components
    v_x = distance_x / delta_t
    v_y = distance_y / delta_t

    #call function to detect which rod needs to be moved
    get_trajectory(ball_pos_2, v_x, v_y)

#Function that determines what rod will need to have its players moved, and the trajectory of the ball
def get_trajectory(ball_pos_2, v_x, v_y):
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

    if abs(v_x) > 1e-6:
        t_intercept = (CURR_ROD_X - ball_pos_2[0]) / v_x
        y_rod_new = ball_pos_2[1] + v_y * t_intercept

    return rod_move, y_rod_new


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


game_state, player_ys = generate_game_state()

