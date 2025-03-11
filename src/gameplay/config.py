"""
config.py: contains constants shared across files
"""

from adafruit_servokit import ServoKit

from adafruit_pca9685 import PCA9685

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

player_ys = [[0,0,0],[0,0,0],[0,0,0]]

myKit=ServoKit(channels=16)
