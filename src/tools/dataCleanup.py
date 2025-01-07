import json 
import os
import shutil

json_path = 'data/labels/labels.json'
source_dir = 'data/image_data/images'
destination_dir = 'data/image_data/images_1'
images_moved = 'data/images_moved.txt'
missed_files = 'data/missing_files.txt'

if not os.path.exists(images_moved):
    with open(images_moved, 'w') as f:
        pass

if not os.path.exists(destination_dir):
    raise FileNotFoundError(f"Destination directory {destination_dir} does not exist.")

def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def save_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def process_and_move(json_path, source_dir, destination_dir):
    new_json = []
    moved_images = []
    missing_files = []
    data = load_json(json_path)
    print("Processing data...")
    for entry in data:
        x = entry.get('x')
        y = entry.get('y')
        image_name = entry.get('image')
        if x > 2303 or y > 1295:
            moved_images.append(image_name)
            source_path = os.path.join(source_dir, image_name)
            destination_path = os.path.join(destination_dir, image_name)
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
            else:
                print(f"Source path {source_path} does not exist.")
                missing_files.append(image_name)

        else:
            new_json.append(entry)
    print("Saving data...")
    with open(images_moved, 'a') as f:
        for image_name in moved_images:
            f.write(f"{image_name}\n")
    print("Saving missing files...")
    with open(missed_files, 'w') as f:
        for image_name in missing_files:
            f.write(f"{image_name}\n")

    save_json(new_json, json_path)

def main():
    print("Running data cleanup...")
    try:
        process_and_move(json_path, source_dir, destination_dir)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
