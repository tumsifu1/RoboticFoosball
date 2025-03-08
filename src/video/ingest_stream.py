import cv2
import numpy as np
import torch
from models.binaryClassifier.model import BinaryClassifier
from models.BallLocalization.model_snoutNetBase import BallLocalization
import torch.nn.functional as F
import time
from torchvision import transforms
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib


# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
classifier = BinaryClassifier().to(device).eval()
localizer = BallLocalization().to(device).eval()

classifier.load_state_dict(torch.load('./src/weights/classifier.pth', map_location=device))
localizer.load_state_dict(torch.load('./src/weights/localizer.pth', map_location=device))

# Preprocessing Pipeline
COMPUTED_MEAN = [0.1249, 0.1399, 0.1198]
COMPUTED_STD = [0.1205, 0.1251, 0.1123]

preprocess = transforms.Compose([
    transforms.Normalize(mean=COMPUTED_MEAN, std=COMPUTED_STD)
])

# Frame queue (ensure only latest frame is processed)
frame_queue = Queue(maxsize=1)

def segment_image_fast(image, grid_size=8):
    """Fast segmentation of a PIL image into grid_size x grid_size tiles."""
    
    # Convert PIL image to NumPy array
    image_np = np.array(image)
    
    # Get dimensions
    H, W, C = image_np.shape  # Ensure it's in H, W, C format
    region_h, region_w = H // grid_size, W // grid_size

    # Extract tiles
    tiles = np.array([
        image_np[y:y+region_h, x:x+region_w]
        for y in range(0, H, region_h) 
        for x in range(0, W, region_w)
    ])
    
    return tiles #area of numpy images shape = (region, region_w, C)


def ingest_stream():
    Gst.init(None)
    """Reads frames from the video stream and processes them using NVIDIA decoder (without CUDA)."""
    def on_new_sample(sink):
        """Callback function for retrieving frames"""
        sample = sink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            width = caps.get_structure(0).get_int("width")[1]
            height = caps.get_structure(0).get_int("height")[1]
            success, map_info = buffer.map(Gst.MapFlags.READ)
            print(width, height, map_info)
            if success:
                print(f"Buffer size: {len(map_info.data)} | Expected: {width * height * 3}")
                
                if len(map_info.data) == width * height * 3:
                    frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3))
                    print("Frame successfully reshaped.")

                    try:
                        frame_queue.put_nowait(frame)  # Send frame to display thread
                    except:
                        print("Queue full")  
                    

                else:
                    print("Skipping frame due to size mismatch.")

                buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK

    pipeline = Gst.parse_launch(
        "udpsrc port=5000 caps=\"application/x-rtp, encoding-name=H264, payload=96\" ! "
        "rtpjitterbuffer latency=10 drop-on-latency=true ! rtph264depay ! "
        "h264parse ! nvv4l2decoder ! nvvidconv ! videoconvert ! "
        "video/x-raw, format=RGB ! appsink name=sink emit-signals=True max-buffers=1 drop=True"
    )


    appsink = pipeline.get_by_name("sink")
    appsink.connect("new-sample", on_new_sample)

    pipeline.set_state(Gst.State.PLAYING)

    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Stopping...")
        pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()

def plot_detections(original_image, detected_tile, detected_positions, grid_size, image_shape):
    """Plot both the original image and detected tile with ball positions."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot full image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image with Absolute Coordinates")

    # Overlay absolute coordinates
    H, W, _ = image_shape
    region_h, region_w = H // grid_size, W // grid_size
    for row, col, prob in detected_positions:
        x_abs = int((col + 0.5) * region_w)  # Center of tile
        y_abs = int((row + 0.5) * region_h)  
        axes[0].scatter(x_abs, y_abs, c='red', marker='o', s=100, label="Ball Position")

    # Plot detected tile
    axes[1].imshow(detected_tile)
    axes[1].set_title("Detected Tile with Relative Coordinates")

    # Overlay relative coordinates (center of tile)
    axes[1].scatter(region_w // 2, region_h // 2, c='blue', marker='o', s=100, label="Ball Position")

    plt.show()

def reconstruct_image(tiles, grid_size, image_shape):
    """Rebuild the original image from tiles."""
    H, W, C = image_shape  # Original image shape
    region_h, region_w = H // grid_size, W // grid_size

    #Reconstruct the full image
    full_image = np.zeros((H, W, C), dtype=np.uint8)

    idx = 0
    for y in range(0, H, region_h):
        for x in range(0, W, region_w):
            full_image[y:y+region_h, x:x+region_w] = tiles[idx]
            idx += 1

    return full_image

def process_frame():
    """Processes frames in real-time with low latency on TX2."""
    while True:
        if frame_queue.empty():
            time.sleep(0.0001)
        else:
            # Assume frame is obtained from frame_queue
            frame = frame_queue.get()  # Original image (PIL or NumPy)

            #Segment Image into Tiles
            grid_size = 8
            tiles = segment_image_fast(frame_np, grid_size)  # NumPy format (H, W, C)

            #Convert tiles to PyTorch Tensors and Normalize
            tiles_tensors = torch.stack([
                preprocess(torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0) 
                for tile in tiles
            ]).to(device)

            #Run batch inference
            with torch.no_grad():
                logits = classifier(tiles_tensors)  # Shape: (batch_size, 1)
                probs = torch.sigmoid(logits).squeeze()  # Convert logits to probabilities
                tiles_tensors = F.interpolate(tiles_tensors, size=(227, 227), mode='bilinear', align_corners=False)
                coordinates = localizer(tiles_tensors)  # Get predicted ball coordinates (relative to tile)
                print(coordinates)

            #Find tiles where the ball is detected
            threshold = 0.5
            ball_tiles = (probs > threshold).nonzero(as_tuple=True)[0].tolist()

            #Convert tile indices to grid positions
            detected_positions = [(idx // grid_size, idx % grid_size, probs[idx].item()) for idx in ball_tiles]
            print(f"Detected position: {detected_positions}")

            #Rebuild original image
            reconstructed_image = reconstruct_image(tiles, grid_size, frame_np.shape)

            # Extract detected tile
            if detected_positions:
                row, col, _ = detected_positions[0]  # Assuming one detection for now
                tile_idx = row * grid_size + col
                detected_tile = tiles[tile_idx]
            else:
                detected_tile = np.zeros_like(tiles[0])  # Blank if no detection

            #  Plot Results
            # plot_detections(reconstructed_image, detected_tile, detected_positions, grid_size, frame_np.shape)

if __name__ == "__main__":
    #todo: uncomment out when ready to ingest
    ingest_process = Process(target=ingest_stream)
    process_process = Process(target=process_frame)

    ingest_process.start()
    process_process.start()

    ingest_process.join()
    process_process.join()


