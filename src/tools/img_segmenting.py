import cv2
import argparse
import os
argParser = argparse.ArgumentParser()
argParser.add_argument("-v", type=str, default='data/videos/test004.mp4', help="directory of source video file")
args = argParser.parse_args()

folder_path = "data/not_labelled/images_notLabelled"
file_path = os.path.join(folder_path)

# Create the folder structure if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

vid = cv2.VideoCapture(args.v)
img_counter = 7702 #last image number+1 in the folder
read, img = vid.read()
while read:
    cv2.imwrite(f"{folder_path}/img_{img_counter}.jpg", img)
    read, img = vid.read()
    img_counter += 1
print(f"Extracted {img_counter} images from video.")