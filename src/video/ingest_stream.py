import numpy as np
import torch
from models.binaryClassifier.model import BinaryClassifier
from models.BallLocalization.model_snoutNetBase import BallLocalization
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize
# from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import time
import cv2

DEBUG = True

# Setup Device
device = torch.device("cuda")

# Load models
classifier = BinaryClassifier().to(device).eval()
localizer = BallLocalization().to(device).eval()

classifier.load_state_dict(torch.load('./src/weights/classifier.pth', map_location=device))
localizer.load_state_dict(torch.load('./src/weights/localizer.pth', map_location=device))

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
                    process_frame(frame)


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

    plt.show(block=False)
    plt.pause(0.001)

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
    total_start = time.time()
    print("Processing Frame")
    
    # Segment Image into Tiles
    segment_start = time.time()
    grid_size = 8
    tiles = segment_image_fast(frame, grid_size)
    segment_end = time.time()
    print(f"Segmentation time: {(segment_end - segment_start) * 1000:.2f} ms")
    
    # Stack tiles and convert to float32
    preproc_start = time.time()
    tiles = np.stack(tiles)  # Stack all tiles into a single array
    tiles = tiles.astype(np.float32)
    if DEBUG:
        print(f"Stacked shape: {tiles.shape}, dtype: {tiles.dtype}")
        print(f"Tiles array memory info: {tiles.nbytes / (1024 * 1024):.2f} MB")
        print(f"CUDA memory before conversion: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")
        
    # Convert to PyTorch tensor
    tiles_tensor = torch.from_numpy(tiles).permute(0, 3, 1, 2).float()
    tensor_end = time.time()
    
    # Normalize
    tiles_tensor = tiles_tensor / 255.0
    
    # Preprocess
    tiles_tensor = preprocess(tiles_tensor)
    tiles_tensor = tiles_tensor.to(device)

    preproc_end = time.time()
    print(f"Tensor Preprocess time: {(preproc_end - preproc_start) * 1000:.2f} ms")
    
    # Run ML models
    with torch.no_grad():
        # Classifier
        classifier_start = time.time()
        logits = classifier(tiles_tensor)
        torch.cuda.synchronize()  # Wait for GPU operations to complete
        probs = torch.sigmoid(logits).squeeze()
        classifier_end = time.time()
        print(f"Classifier time: {(classifier_end - classifier_start) * 1000:.2f} ms")
        print(f"Logits:\n{logits}\bProbs:\n{probs}")
        # Resize for localizer
        localizer_start = time.time()
        tiles_tensor = F.interpolate(tiles_tensor, size=(227, 227), mode='bilinear', align_corners=False)
        
        # Run localizer
        coordinates = localizer(tiles_tensor)
        torch.cuda.synchronize()  # Wait for GPU operations to complete
        localizer_end = time.time()
        print(f"Localizer time: {(localizer_end - localizer_start) * 1000:.2f} ms")
    
    # Detection logic
    detect_start = time.time()
    threshold = 0.99
    ball_tiles = (probs > threshold).nonzero(as_tuple=True)[0].tolist()
    detected_positions = [(idx // grid_size, idx % grid_size, probs[idx].item()) for idx in ball_tiles]
    detect_end = time.time()
    print(f"Detection logic time: {(detect_end - detect_start) * 1000:.2f} ms")
    
    # Reconstruction
    recon_start = time.time()
    reconstructed_image = reconstruct_image(tiles, grid_size, frame.shape)
    
    # Extract detected tile
    if detected_positions:
        row, col, _ = detected_positions[0]
        tile_idx = row * grid_size + col
        H, W, _ = frame.shape
        region_h, region_w = H // grid_size, W // grid_size
        y = row * region_h
        x = col * region_w
        detected_tile = reconstructed_image[y:y+region_h, x:x+region_w]

    else:
        detected_tile = np.zeros_like(tiles[0])
    recon_end = time.time()
    print(f"Reconstruction time: {(recon_end - recon_start) * 1000:.2f} ms")
    
    # Calculate total time
    total_end = time.time()
    total_time = total_end - total_start
    print(f"TOTAL FRAME PROCESSING TIME: {total_time * 1000:.2f} ms")
    
    # Plot if debugging is enabled
    if DEBUG:
        plot_detections(reconstructed_image, detected_tile, detected_positions, grid_size, frame.shape)

if __name__ == "__main__":
    #todo: uncomment out when ready to ingest
    # ingest_process = Process(target=ingest_stream)
    # process_process = Process(target=process_frame)

    # ingest_process.start()
    # process_process.start()

    # ingest_process.join()
    # process_process.join()

    ingest_stream()
