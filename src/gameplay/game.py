import numpy as np
import pandas as pd

#Dimensions of elements of the tables, pixel-wise
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

#INITIALLY MEASURED HOWEVER WILL NEED TO BE CENTERED EVEN BETTER
TABLE_LEFT = 172
TABLE_RIGHT = FRAME_WIDTH - 172
TABLE_TOP = 127
TABLE_BOTTOM = FRAME_HEIGHT - 127
TABLE_HEIGHT = TABLE_BOTTOM - TABLE_TOP

#Divisions in between thirds of the table. Player will be chosen depending on which zone the ball will go to
TABLE_DIV_1_Y = TABLE_TOP + TABLE_HEIGHT // 3
TABLE_DIV_2_Y = TABLE_TOP + 2 * TABLE_HEIGHT // 3

#ROD LOCATIONS (+8)
LEFT_ROD_X = 425
MIDDLE_ROD_X = 708
RIGHT_ROD_X = 991

#PLAYER DIMS:
PLAYER_WIDTH = 36
PLAYER_GAP = 138
HALF_PLAYER = 37 // 2

#PLAYER BOUNDS
PLAYER_UPPER = TABLE_TOP + 19
PLAYER_LOWER = TABLE_BOTTOM - 19

def generate_game_state(frame_width=1280, frame_height=720):
    # -1 OOB area
    game_state = np.full((frame_height, frame_width), -1, dtype=int)
    
    #table surface is 0s
    game_state[TABLE_TOP:TABLE_BOTTOM, TABLE_LEFT:TABLE_RIGHT] = 0
    
    #Set rod positions to 1s, players are 2s
    rod_positions = [LEFT_ROD_X, MIDDLE_ROD_X, RIGHT_ROD_X]
    player_start_y = TABLE_TOP + 19  # 1st player cntr

    initial_player_y = [[0,0,0],[0,0,0],[0,0,0]]
    
    for j, rod_x in enumerate(rod_positions):
        if TABLE_LEFT <= rod_x < TABLE_RIGHT:
            game_state[TABLE_TOP:TABLE_BOTTOM, rod_x] = 1
            # #3 players
            # for i in range(3):
            #     player_y = player_start_y + i * PLAYER_GAPS
            #     game_state[max(TABLE_TOP, player_y - HALF_PLAYER): min(TABLE_BOTTOM, player_y + HALF_PLAYER + 1), rod_x] = 2
            #     game_state[player_y, rod_x] = 3  #Centre of players as 3s. May remove player widths completely

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
    
    # euc_distance = np.sqrt(distance_x**2 + distance_y**2)

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

def motor_drive(rod_move, movement_amount):
    print(f"Moving rod {rod_move} by {movement_amount:.2f} pixels.")
    #Here the code will be to convert pixels into rotations of motor, etc
        #will also implement the array changing within this
        #another place where the state array will change is within the trajectory mapping function
        #as it will have the new ball locations, only question is how often will it be run?
        #every 2 frames, every few?

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

    #how much rod needs to move
    movement_amount = y_rod_new - current_player_y

    print(f"Player {player_index + 1} on rod {rod_move} moving from {current_player_y} to {y_rod_new:.2f}")

    motor_drive(rod_move, movement_amount)

    # #update player pos in array
    # player_ys[rod_move][player_index] = y_rod_new // could update this and game array as motor moves

game_state, player_ys = generate_game_state()

