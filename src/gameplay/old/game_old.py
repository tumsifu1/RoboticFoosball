import numpy as np

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

LEFT_ROD_X = 500        
MIDDLE_ROD_X = 1000
RIGHT_ROD_X = 1500

def get_trajectory(ball_pos_1, ball_pos_2):
    distance_x = ball_pos_2[0] - ball_pos_1[0]
    distance_y = ball_pos_2[1] - ball_pos_1[1]
    delta_t = ball_pos_2[2] - ball_pos_1[2]

    if delta_t == 0:
        return

    euc_distance = np.sqrt(distance_x**2 + distance_y**2)
    angle = np.degrees(np.arctan2(distance_y, distance_x))

    v_x = distance_x / delta_t
    v_y = distance_y / delta_t
    velocity = euc_distance / delta_t

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
        trigger_motor(rod_move, y_rod_new)

def trigger_motor(rod_move, y_rod_new):
    if rod_move == -1:
        print("no horizontal movement\n")
    else:
        print("Arm Motor ", rod_move, " Triggered\n")
        trigger_shot(rod_move, y_rod_new)

def trigger_shot(rod_move, y_rod_new):
    print("Shot Motor ", rod_move, " Triggered\n")

get_trajectory((400, 100, 7000), (600, 130, 7020))
