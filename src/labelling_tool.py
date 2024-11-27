import cv2
import json
import os
import argparse
import shutil

def move_image(image_dir, output_dir, image_name):
    """Move an image from one directory to another."""
    shutil.move(os.path.join(image_dir, image_name), os.path.join(output_dir, image_name))

def mouse_click(event, x, y, flags, param):
    """Handle mouse clicks to label the ball in an image."""
    labeled_data, args, image_files, img_index = param 
    if event == cv2.EVENT_LBUTTONDOWN:
        image_name = image_files[img_index[0]]
        labeled_data.append({"image": image_name, "ball_exists": 1, "x": x, "y": y})
        move_image(args.image_dir, args.output_dir, image_name)
        process_next_image(args, labeled_data, image_files, img_index)


def skip_image(args, labeled_data, image_files, img_index):
    """Skip the current image, marking it as no ball in the frame."""
    image_name = image_files[img_index[0]]
    labeled_data.append({"image": image_name, "ball_exists": 0})
    move_image(args.image_dir, args.output_dir, image_name)
    print(f"Image {image_name} skipped.")
    process_next_image(args, labeled_data, image_files, img_index)


def discard_image(args, labeled_data, image_files, img_index):
    """Discard the current image without labeling or moving it."""
    image_name = image_files[img_index[0]]
    print(f"Image {image_name} discarded.")
    process_next_image(args, None, image_files, img_index)

def write_to_json(data, file):
    if data is not None:
        with open(file, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {file}.")

def process_next_image(args, labeled_data, image_files, img_index):
    """Load the next image for labeling or finish if all images are processed."""
    img_index[0] += 1
    if img_index[0] < len(image_files):
        image_name = image_files[img_index[0]]
        img = cv2.imread(os.path.join(args.image_dir, image_name))
        cv2.imshow("Labeling Tool", img)
    else:
        write_to_json(labeled_data, args.output_json)
        cv2.destroyAllWindows()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Image Labeling Tool with Discard Feature")
    parser.add_argument("--image_dir", required=True, help="Directory with images to label")
    parser.add_argument("--output_dir", required=True, help="Directory to save labeled images")
    parser.add_argument("--output_json", required=True, help="Path to save JSON file with labels")
    args = parser.parse_args()

    # Initialize variables
    labeled_data = []
    img_index = [-1]  # starts at negative one to start at 0 after incrementing || img_index is a list to allow for mutable variables 
    image_files = [f for f in os.listdir(args.image_dir) if f.endswith(".jpg") or f.endswith(".png")]

    if not image_files:
        print("No images found in the specified directory.")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set up OpenCV window and mouse callback
    cv2.namedWindow("Labeling Tool")
    cv2.setMouseCallback("Labeling Tool", mouse_click, param=(labeled_data, args, image_files, img_index))

    print("Instructions:")
    print("- Left Click: Label ball on image")
    print("- Press 'd': Discard image (not saved or labeled, stays in original directory)")
    print("- Press 's': Skip image (labeled as no ball in frame)")
    print("- Press 'q': Quit")

    # Start labeling
    process_next_image(args, labeled_data, image_files, img_index)

    # Listen for key presses
    while True:
        key = cv2.waitKey(0) & 0xFF # mask 0xFF to get the last 8 bits
        if key == ord('d'):  # Discard image
            discard_image(args, labeled_data, image_files, img_index)
        elif key == ord('s'):  # Skip image
            skip_image(args, labeled_data, image_files, img_index)
        elif key == ord('q'):  # Quit
            write_to_json(labeled_data, args.output_json)
            print("Exiting labeling tool.")
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
