"""
config.py: contains constants shared across files
"""


# from adafruit_pca9685 import PCA9685

import busio
import board


from adafruit_servokit import ServoKit


#Dimensions of elements of the tables, pixel-wise
FRAME_WIDTH = 2304
FRAME_HEIGHT = 1296

#PLAYER DIMS:
PLAYER_WIDTH = 65
WASHER_THICKNESS = 12           #Bounds will be needed to be added for inferencing or an adjustment to just push the y value of the ball within bounds
PLAYER_GAP = 258
HALF_PLAYER = PLAYER_WIDTH // 2

#INITIALLY MEASURED
TABLE_LEFT = 272
TABLE_RIGHT = FRAME_WIDTH - 318
TABLE_TOP = 208 + WASHER_THICKNESS
TABLE_BOTTOM = FRAME_HEIGHT - 236
TABLE_HEIGHT = TABLE_BOTTOM - TABLE_TOP
TABLE_WIDTH = TABLE_RIGHT - TABLE_LEFT

#Divisions in between thirds of the table. Player will be chosen depending on which zone the ball will go to
TABLE_DIV_1_Y = TABLE_TOP + TABLE_HEIGHT // 3
TABLE_DIV_2_Y = TABLE_TOP + 2 * TABLE_HEIGHT // 3

#ROD LOCATIONS
LEFT_ROD_X = 480
MIDDLE_ROD_X = 1000
RIGHT_ROD_X = 1518

#Minimum movement to prevent trigger motor to be called if it is too small (prevent unnecessary start/stops)
MIN_MOVEMENT = PLAYER_WIDTH//2

player_ys = [[218, 476, 734], [218, 476, 734], [218, 476, 734]]

myKit=ServoKit(channels=16)


