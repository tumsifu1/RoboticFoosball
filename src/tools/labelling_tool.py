import cv2
import json
import os
import argparse
import shutil


def move_image(image_dir, output_dir, image_name):
    """Move an image from one directory to another."""
    shutil.move(os.path.join(image_dir, image_name), os.path.join(output_dir, image_name))

def mouse_click(event, x, y, flags, param):
    """Handle mouse clicks and zooming."""
    labeled_data, args, image_files, img_index, zoom_state = param

    if event == cv2.EVENT_LBUTTONDOWN:
        # Adjust coordinates based on zoom and crop
        zoom_level = zoom_state["level"]
        zoom_factor = 1 + 0.2 * zoom_level
        crop_x, crop_y = zoom_state.get("crop_offset", (0, 0))  # Crop offsets
        original_x = int((x + crop_x) / zoom_factor)
        original_y = int((y + crop_y) / zoom_factor)

        # Label the ball using original image coordinates
        image_name = image_files[img_index[0]]
        labeled_data.append({"image": image_name, "ball_exists": 1, "x": original_x, "y": original_y})
        move_image(args.image_dir, args.output_dir, image_name)

        process_next_image(args, labeled_data, image_files, img_index, zoom_state)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Zoom in where the right click occurs
        zoom_state["level"] = min(zoom_state["level"] + 1, 20)  # Cap zoom level at 20
        zoom_state["center"] = (x, y)  # Update the zoom center to the right-click position
        print(f"Zoom in at ({x}, {y}), Level: {zoom_state['level']}")
        update_zoom(args, image_files, img_index, zoom_state)

def update_zoom(args, image_files, img_index, zoom_state):
    """Update the zoom level and redisplay the image."""
    image_name = image_files[img_index[0]]
    img = cv2.imread(os.path.join(args.image_dir, image_name))
    zoom_level = zoom_state["level"]

    zoom_factor = 1 + 0.2 * zoom_level
    h, w = img.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)

    # Resize the image based on zoom level
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Determine the cropping region centered on the click position
    center_x, center_y = zoom_state.get("center", (w // 2, h // 2))  # Default to center if no click
    center_x = int(center_x * zoom_factor)
    center_y = int(center_y * zoom_factor)

    crop_x1 = max(center_x - w // 2, 0)
    crop_y1 = max(center_y - h // 2, 0)
    crop_x2 = min(crop_x1 + w, new_w)
    crop_y2 = min(crop_y1 + h, new_h)

    # Store the offsets for coordinate mapping
    zoom_state["crop_offset"] = (crop_x1, crop_y1)

    # Crop the image
    img_cropped = img_resized[crop_y1:crop_y2, crop_x1:crop_x2]
    zoom_state["current_image"] = img_cropped

    cv2.imshow("Labeling Tool", img_cropped)

def skip_image(args, labeled_data, image_files, img_index, zoom_state):
    """Skip the current image, marking it as no ball in the frame."""
    image_name = image_files[img_index[0]]
    labeled_data.append({"image": image_name, "ball_exists": 0})
    move_image(args.image_dir, args.output_dir, image_name)
    print(f"Image {image_name} skipped.")
    process_next_image(args, labeled_data, image_files, img_index, zoom_state)

def discard_image(args, labeled_data, image_files, img_index, zoom_state):
    """Discard the current image without labeling or moving it."""
    image_name = image_files[img_index[0]]
    print(f"Image {image_name} discarded.")
    process_next_image(args, None, image_files, img_index, zoom_state)

def write_to_json(data, file):
    if data is not None:
        with open(file, "a") as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {file}.")

def process_next_image(args, labeled_data, image_files, img_index, zoom_state):
    """Load the next image for labeling or finish if all images are processed."""
    img_index[0] += 1
    if img_index[0] < len(image_files):
        image_name = image_files[img_index[0]]
        img = cv2.imread(os.path.join(args.image_dir, image_name))
        zoom_state["current_image"] = img
        zoom_state["level"] = 0  # Reset zoom for the next image
        zoom_state["center"] = None  # Reset zoom center
        cv2.imshow("Labeling Tool", img)
    else:
        write_to_json(labeled_data, args.output_json)
        cv2.destroyAllWindows()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Image Labeling Tool with Zoom Feature")
    parser.add_argument("--image_dir", required=True, help="Directory with images to label")
    parser.add_argument("--output_dir", required=True, help="Directory to save labeled images")
    parser.add_argument("--output_json", required=True, help="Path to save JSON file with labels")
    args = parser.parse_args()

    # Initialize variables
    labeled_data = []
    img_index = [-1]  # Mutable index for the current image
    image_files = [f for f in os.listdir(args.image_dir) if f.endswith(".jpg") or f.endswith(".png")]
    zoom_state = {"level": 0, "center": None, "current_image": None}  # Track zoom level and current image

    cv2.namedWindow("Labeling Tool")
    cv2.setMouseCallback("Labeling Tool", mouse_click, param=(labeled_data, args, image_files, img_index, zoom_state))

    if not image_files:
        print("No images found in the specified directory.")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("Instructions:")
    print("- Left Click: Label ball on image")
    print("- Right Click: Zoom in at clicked location")
    print("- Space: Zoom out")
    print("- Press 'd': Discard image (not saved or labeled, stays in original directory)")
    print("- Press 's': Skip image (labeled as no ball in frame)")
    print("- Press 'q': Quit")

    # Start labeling
    process_next_image(args, labeled_data, image_files, img_index, zoom_state)

    # Listen for key presses
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord(' '):  # Spacebar to zoom out
            zoom_state["level"] = max(zoom_state["level"] - 1, 0)  # Cap zoom level at 0
            #print(f"Zoom level (out): {zoom_state['level']}")
            update_zoom(args, image_files, img_index, zoom_state)
        elif key == ord('d'):  # Discard image
            discard_image(args, labeled_data, image_files, img_index, zoom_state)
        elif key == ord('s'):  # Skip image
            skip_image(args, labeled_data, image_files, img_index, zoom_state)
        elif key == ord('q'):  # Quit
            write_to_json(labeled_data, args.output_json)
            print("Exiting labeling tool.")
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
