import numpy as np
import torch
from models.binary_classifier import BinaryClassifier
from models.localizer import BallLocalization
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize
# from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import time
import argparse

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

preprocess = Compose([
    Normalize(mean=COMPUTED_MEAN, std=COMPUTED_STD)
])

# frame_queue = Queue(maxsize=1)

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
        "h264parse ! queue max-size-buffers=3 leaky=downstream ! nvv4l2decoder enable-max-performance=1 disable-dpb=true "
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
    axes[1].scatter(region_w // 2, region_h // 2, c='red', marker='o', s=100, label="Ball Position")

    fig.savefig("./saved_frames/plot.jpg")
    quit()

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

def process_frame(frame):
    """Processes frames in real-time with low latency on TX2."""
    if TIMING:
        total_start = time.time()
    if DEBUG:
        print("Processing Frame")
    
    # Segment Image into Tiles
    segment_start = time.time()
    grid_size = 8
    tiles = segment_image_fast(frame, grid_size)
    if TIMING:
        segment_end = time.time()
        print(f"Segmentation time: {(segment_end - segment_start) * 1000:.2f} ms")
    
    if TIMING:
        preproc_start = time.time()
    tiles = np.stack(tiles)  # Stack all tiles into a single array
    tiles = tiles.astype(np.float32)
    if DEBUG:
        print(f"Stacked shape: {tiles.shape}, dtype: {tiles.dtype}")
        print(f"Tiles array memory info: {tiles.nbytes / (1024 * 1024):.2f} MB")
        print(f"CUDA memory before conversion: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")
        
    # Convert to PyTorch tensor
    tiles_tensor = torch.from_numpy(tiles).permute(0, 3, 1, 2).float()
    if TIMING:
        tensor_end = time.time()
    
    # Normalize
    tiles_tensor = tiles_tensor / 255.0
    
    # Preprocess
    tiles_tensor = preprocess(tiles_tensor)
    tiles_tensor = tiles_tensor.to(device)

    if TIMING:
        preproc_end = time.time()
        print(f"Tensor Preprocess time: {(preproc_end - preproc_start) * 1000:.2f} ms")
    
    # Run ML models
    with torch.no_grad():
        # Classifier
        if TIMING:
            classifier_start = time.time()

        logits = classifier(tiles_tensor)

        if TIMING:
            cl_model_end = time.time()
            print(f"Classifier Model Ran in: {(cl_model_end - classifier_start) * 1000:.2f} ms")

        probs = torch.sigmoid(logits).squeeze()

        if TIMING:
            classifier_end = time.time()
            print(f"Sigmoid time: {(classifier_end - cl_model_end) * 1000:.2f} ms")
            print(f"Total classifier time: {(classifier_end - classifier_start) * 1000:.2f} ms")
       
        # if DEBUG:
            # print(f"Logits:\n{logits}\bProbs:\n{probs}")

        if TIMING:
            localizer_start = time.time()

        # tiles_tensor = F.interpolate(tiles_tensor, size=(227, 227), mode='bilinear', align_corners=False)
        
        # Run localizer
        coordinates = localizer(tiles_tensor)
        if TIMING:
            loc_model_end = time.time()
            print(f"Localizer Model Ran Time: {(loc_model_end - localizer_start) * 1000:.2f} ms")

    # Detection logic
    if TIMING:
        detect_start = time.time()

    threshold = 0.9
    ball_tiles = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
    detected_positions = [(idx // grid_size, idx % grid_size, probs[idx].item()) for idx in ball_tiles]
    
    if TIMING:
        detect_end = time.time()
        print(f"Detection logic time: {(detect_end - detect_start) * 1000:.2f} ms")
    print(f"Detected Positions: {detected_positions}")
    
    # Reconstruction
    if TIMING:
        recon_start = time.time()
    
    if PLOT:
        reconstructed_image = reconstruct_image(tiles, grid_size, frame.shape)
    
    # Extract detected tile
    if detected_positions:
        row, col, _ = detected_positions[0]
        tile_idx = row * grid_size + col
        H, W, _ = frame.shape
        region_h, region_w = H // grid_size, W // grid_size
        y = row * region_h
        x = col * region_w
        if PLOT:
            detected_tile = reconstructed_image[y:y+region_h, x:x+region_w]
        print(f"Global Prediction: ({x, y})")
    else:
        detected_tile = np.zeros_like(tiles[0])

    if TIMING:
        recon_end = time.time()
        print(f"Reconstruction time: {(recon_end - recon_start) * 1000:.2f} ms")
    if PLOT:
        plot_detections(reconstructed_image, detected_tile, detected_positions, grid_size, frame.shape)

    if TIMING:
        total_end = time.time()
        total_time = total_end - total_start
        print(f"TOTAL FRAME PROCESSING TIME: {total_time * 1000:.2f} ms")
    
    return (x, y) if detected_positions else None

if __name__ == "__main__":
    global DEBUG, TIMING, PLOT
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", default='T', choices=['T','F'], help="Enable debug logs")
    parser.add_argument("-t", default='F', choices=['T','F'], help="Enable timing logs")
    parser.add_argument("-p", default='F', choices=['T','F'], help="Plot one output")
    
    args = parser.parse_args()
    
    DEBUG = (args.d == 'T')
    TIMING = (args.t == 'T')
    PLOT = (args.p == 'T')
    ingest_stream()
