from config import *
import numpy as np

def game_init():
    #Set rod positions to 1s, players are 2s
    rod_positions = [LEFT_ROD_X, MIDDLE_ROD_X, RIGHT_ROD_X]
    player_start_y = TABLE_TOP + HALF_PLAYER  # 1st player cntr

    player_ys = [[0,0,0],[0,0,0],[0,0,0]]

    for j, rod_x in enumerate(rod_positions):
        if TABLE_LEFT <= rod_x < TABLE_RIGHT:
            #3 players
            for i in range(3):
                player_y = player_start_y + i * PLAYER_GAP
                player_ys[j][i] = player_y

    print(player_ys)

    # return 


game_init()