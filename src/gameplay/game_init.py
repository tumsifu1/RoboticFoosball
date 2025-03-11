from config import *
import numpy as np

def game_init():
    #Set rod positions to 1s, players are 2s
    rod_positions = [LEFT_ROD_X, MIDDLE_ROD_X, RIGHT_ROD_X]
    player_start_y = TABLE_TOP + HALF_PLAYER  # 1st player cntr

    initial_player_y = [[0,0,0],[0,0,0],[0,0,0]]

    for j, rod_x in enumerate(rod_positions):
        if TABLE_LEFT <= rod_x < TABLE_RIGHT:
            #3 players
            for i in range(3):
                player_y = player_start_y + i * PLAYER_GAP
                initial_player_y[j][i] = player_y

    ball_old = (-1, -1, -1)
    ball_cur = (-1, -1, -1)

    return initial_player_y, ball_old, ball_cur