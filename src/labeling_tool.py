import cv2
import json
import os
import argparse
from datetime import datetime

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Image Labeling Tool")
parser.add_argument("--image_dir", required=True, help="Directory with images to label")
parser.add_argument("--output_dir", required=True, help="Directory to save labeled images")
parser.add_argument("--output_json", required=True, help="Path to save JSON file with labels")
args = parser.parse_args()

# Initialize labeled data list
labeled_data = []

# Ensure output directory exists
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# List of images to process
image_files = [f for f in os.listdir(args.image_dir) if f.endswith(".jpg") or f.endswith(".png")]
if not image_files:
    print("No images found in the specified directory.")
    exit()

# Initialize counters
img_index = 0
image_name = ""
img = None

# Function to handle mouse clicks
def mouse_click(event, x, y, flags, param):
    global labeled_data, img, img_index, image_name

    if event == cv2.EVENT_LBUTTONDOWN:
        # Save the coordinates for the center point
        labeled_data.append({"image": image_name, "x": x, "y": y})

        # Mark the click position on the image for reference
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Labeling Tool", img)

        # Save the labeled image with the click marked
        cv2.imwrite(f"{args.output_dir}/labeled_{image_name}", img)

        # Move to the next image
        process_next_image()

# Function to load the next image
def process_next_image():
    global img, img_index, image_name

    if img_index < len(image_files):
        image_name = image_files[img_index]
        img = cv2.imread(os.path.join(args.image_dir, image_name))
        cv2.imshow("Labeling Tool", img)
        img_index += 1
    else:
        # Save labeled data to JSON
        with open(args.output_json, "w") as f:
            json.dump(labeled_data, f, indent=4)
        print(f"Labeling completed. Data saved to {args.output_json}.")
        cv2.destroyAllWindows()

# Set up OpenCV window and mouse callback
cv2.namedWindow("Labeling Tool")
cv2.setMouseCallback("Labeling Tool", mouse_click)

# Load the first image
process_next_image()

# Keep the window open until all images are labeled
cv2.waitKey(0)
cv2.destroyAllWindows()
