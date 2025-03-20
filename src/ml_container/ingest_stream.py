import numpy as np
import torch
from models.binary_classifier import BinaryClassifier
from models.localizer import BallLocalization
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize
import matplotlib.pyplot as plt
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import time
import argparse

global DEBUG, TIMING, PLOT
DEBUG, TIMING, PLOT = False, False, False

# Setup Device
device = torch.device("cuda")

# Load models
classifier = BinaryClassifier().to(device).eval()
localizer = BallLocalization().to(device).eval()

classifier.load_state_dict(torch.load('./weights/classifier.pth', map_location=device))
localizer.load_state_dict(torch.load('./weights/localizer.pth', map_location=device))

# Preprocessing Pipeline
COMPUTED_MEAN = [0.1249, 0.1399, 0.1198]
COMPUTED_STD = [0.1205, 0.1251, 0.1123]

mean_tensor = torch.tensor(COMPUTED_MEAN, device=device).view(1,3,1,1)
std_tensor = torch.tensor(COMPUTED_STD, device=device).view(1,3,1,1)

def preprocess(tensor):
    return (tensor - mean_tensor) / std_tensor


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
            if success:
                if len(map_info.data) == width * height * 3:
                    frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3))
                    coords = process_frame(frame)
                buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK

    pipeline = Gst.parse_launch(
        "udpsrc port=5000 buffer-size=200000 caps=\"application/x-rtp, encoding-name=H264, payload=96\" ! "
        "rtpjitterbuffer latency=50 drop-on-latency=true do-lost=true ! rtph264depay ! "
        "h264parse ! nvv4l2decoder enable-max-performance=1 disable-dpb=true "
        " ! nvvidconv ! videoconvert ! video/x-raw, format=RGB ! appsink name=sink emit-signals=True max-buffers=1 drop=True"
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


def visualize_prediction(frame, global_position):
    """
    Visualize the prediction on the original frame and save the image.
    
    Args:
    frame (numpy.ndarray): The original frame (H x W x 3)
    global_position (tuple): (x, y) coordinates of the prediction
    save_path (str): Path to save the output image
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Display the original frame
    ax.imshow(frame)
    
    # Mark the prediction
    x, y = global_position
    ax.scatter(x, y, c='red', s=100, marker='x', linewidths=2)
    
    # Add a circle around the prediction for better visibility
    circle = plt.Circle((x, y), 20, fill=False, edgecolor='red', linewidth=2)
    ax.add_artist(circle)
    
    # Set title and remove axis ticks
    ax.set_title("Foosball Detection")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save the figure
    plt.savefig('./saved_frames/plot.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory

def process_frame(frame):
    frame_tensor = torch.tensor(frame, device=device).unsqueeze(0).permute(0, 3, 1, 2)
    start = time.time()

    # Assume frame_tensor is already on GPU
    grid_size = 8
    H, W = frame_tensor.shape[2], frame_tensor.shape[3]
    region_h, region_w = H // grid_size, W // grid_size

    # Segment the frame into tiles on GPU
    tiles = frame_tensor.unfold(2, region_h, region_h).unfold(3, region_w, region_w)
    tiles = tiles.contiguous().view(-1, 3, region_h, region_w) / 255

    tiles = preprocess(tiles)

    pre_end = time.time()
    print(f"Preprocess: {(pre_end - start) * 1000} ms")

    # Run classifier and localizer
    with torch.no_grad():
        logits = classifier(tiles)
        output = localizer(tiles)
    
    models_end = time.time()
    print(f"Models: {(models_end - pre_end) * 1000} ms")

    # Process results on GPU
    probs = torch.sigmoid(logits).squeeze()
    max_prob_idx = probs.argmax()
    max_prob_val = probs[max_prob_idx].item()  # Get the actual probability value
    # Apply threshold
    probability_threshold = 0.9
    if max_prob_val < probability_threshold:
        return None  # No detection above the threshold

    row = max_prob_idx // grid_size
    col = max_prob_idx % grid_size
    pred_x, pred_y = output[max_prob_idx, 0], output[max_prob_idx, 1]

    res_end = time.time()
    print(f"Process Results: {(res_end - models_end) * 1000} ms")

    pred_x_cpu, pred_y_cpu = pred_x.cpu().item(), pred_y.cpu().item()

    # Scale the localizer's output to the tile size
    norm_x = pred_x / frame_width
    norm_y = pred_y / frame_height
    
    # Scale to tile size
    local_x = norm_x * region_w
    local_y = norm_y * region_h
    
    # Add global offset
    global_x = (col * region_w) + local_x
    global_y = (row * region_h) + local_y

    calc_end = time.time()
    print(f"Calculate Global: {(calc_end - res_end) * 1000} ms")
    print(f"TOTAL: {(calc_end - start) * 1000} ms")

    visualize_prediction(frame, (global_x, global_y))
    quit()
    # Only transfer final coordinates to CPU
    return global_x, global_y

if __name__ == "__main__":
#     global DEBUG, TIMING, PLOT
#     parser = argparse.ArgumentParser()

#     parser.add_argument("-d", default='F', choices=['T','F'], help="Enable debug logs")
#     parser.add_argument("-t", default='F', choices=['T','F'], help="Enable timing logs")
#     parser.add_argument("-p", default='F', choices=['T','F'], help="Plot one output")
    
#     args = parser.parse_args()
    
#     DEBUG = (args.d == 'T')
#     TIMING = (args.t == 'T')
#     PLOT = (args.p == 'T')
    ingest_stream()
