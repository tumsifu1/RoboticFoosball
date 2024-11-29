import cv2
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-v", type=str, default='../data/test.mp4', help="directory of source video file")
args = argParser.parse_args()

vid = cv2.VideoCapture(args.v)
img_counter = 0
read, img = vid.read()
while read:
    cv2.imwrite(f"../data/images_{(img_counter%3) + 1}/img_{img_counter}.jpg", img)
    read, img = vid.read()
    img_counter += 1
