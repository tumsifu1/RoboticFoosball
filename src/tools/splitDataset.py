import os
import shutil
import random
import json

IMAGES_DIR = "data/images"
LABELS_FILE = "data/labels/labels.json"
OUTPUT_DIR = "data"  # will create train/, val/, test/ here

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, \
    "Train/Val/Test ratios must sum to 1.0"

for split_name in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split_name, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split_name, "labels"), exist_ok=True)

with open(LABELS_FILE, "r") as f:
    raw_labels = json.load(f)  # raw_labels is a list
    # Convert the list to a dictionary: {"img_145.jpg": {...}, "img_146.jpg": {...}}
    labels_data = {entry["image"]: {k: v for k, v in entry.items() if k != "image"} for entry in raw_labels}

all_images = list(labels_data.keys())

random.shuffle(all_images)
num_images = len(all_images)
train_count = int(num_images * TRAIN_RATIO)
val_count = int(num_images * VAL_RATIO)
test_count = num_images - train_count - val_count

train_imgs = all_images[:train_count]
val_imgs   = all_images[train_count : train_count + val_count]
test_imgs  = all_images[train_count + val_count:]


def gather_split_data(image_list):
    """Return a dict of labels (just for this split)
       and copy corresponding images to the split folder.
    """
    split_labels = {}
    for img_name in image_list:
        # Copy the image if it exists
        src_img_path = os.path.join(IMAGES_DIR, img_name)
        dst_img_path = os.path.join(OUTPUT_DIR, split_name, "images", img_name)
        
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)

        # Extract labels for this image from the master labels_data
        if img_name in labels_data:
            split_labels[img_name] = labels_data[img_name]

    return split_labels


for split_name, image_list in zip(["train", "val", "test"],
                                  [train_imgs, val_imgs, test_imgs]):
    split_labels = gather_split_data(image_list)
    # Write these labels into a new labels.json inside the split folder
    split_labels_path = os.path.join(OUTPUT_DIR, split_name, "labels", "labels.json")
    with open(split_labels_path, "w") as f:
        json.dump(split_labels, f, indent=2)

print("SPLIT SUMMARY:")
print(f"  Total images: {num_images}")
print(f"  Train: {len(train_imgs)}  | Val: {len(val_imgs)}  | Test: {len(test_imgs)}")
print("Done!")
