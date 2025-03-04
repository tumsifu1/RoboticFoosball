import cv2
import numpy as np
import torch
from models.binaryClassifier.model import BinaryClassifier
from models.BallLocalization.model_snoutNetBase import BallLocalization
from src.tools import vision_utils
import torch.nn.functional as F
import time
from torchvision import transforms
from queue import Queue
from multiprocessing import Process

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
    transforms.ToTensor(),
    transforms.Normalize(mean=COMPUTED_MEAN, std=COMPUTED_STD)
])

# Frame queue (ensure only latest frame is processed)
frame_queue = Queue(maxsize=1)

def segment_image_fast(image, grid_size=8):
    """Fast segmentation using NumPy slicing (OpenCV format: H, W, C)."""
    H, W, _ = image.shape
    region_h, region_w = H // grid_size, W // grid_size

    tiles = [
        image[y:y+region_h, x:x+region_w] 
        for y in range(0, H, region_h) 
        for x in range(0, W, region_w)
    ]
    
    return tiles


def ingest_stream():
    """Reads frames from the video stream and processes them using NVIDIA decoder (without CUDA)."""
    gst_str = (
        "udpsrc port=5000 caps=\"application/x-rtp, encoding-name=H264, payload=96\" ! "
        "rtpjitterbuffer latency=50 drop-on-latency=true mode=1 ! "
        "rtph264depay ! h264parse ! nvv4l2decoder ! "
        "nvvidconv ! appsink"
    )

    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Failed to open video stream.")
        return

    while True:
        read_frame_time = time.time()
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame.")
            break

        # Convert BGR to RGB using NumPy (faster than OpenCV for Jetson TX2)
        frame = frame[:, :, ::-1]  # Efficient BGR to RGB conversion

        # Put frame in the queue (non-blocking, keeps only latest frame)
        if not frame_queue.full():
            frame_queue.put(frame)
        cv2.imshow("Jetson", frame)
        cv2.waitKey(1)
        next_frame_time = time.time()
        print(f"Frame processed in: {(next_frame_time - read_frame_time) * 1000:.2f} ms")

    cap.release()
    cv2.destroyAllWindows()


def process_frame():
    """Processes frames in real-time with low latency on TX2."""
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Convert OpenCV (H, W, C) to (C, H, W) and normalize
            tiles = segment_image_fast(frame)  # NumPy format (H, W, C)

            # Convert to PyTorch tensors and apply preprocessing (efficient batch processing)
            tiles_tensors = torch.stack([
                preprocess(torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0) 
                for tile in tiles
            ]).to(device)

            # Run batch inference (entire grid at once)
            with torch.no_grad():
                logits = classifier(tiles_tensors)  # Shape: (batch_size, 1)
                probs = torch.sigmoid(logits).squeeze()

            # Get tiles that contain the ball
            threshold = 0.5
            ball_tiles = (probs > threshold).nonzero(as_tuple=True)[0].tolist()

            # Convert to grid positions
            grid_size = 8
            detected_positions = [(idx // grid_size, idx % grid_size, probs[idx].item()) for idx in ball_tiles]
            print(f"Detected position: {detected_positions}")
            # Print ball positions
            if detected_positions:
                print(f"Ball detected in tiles: {detected_positions}")
            else:
                print("No ball detected.")


if __name__ == "__main__":
    # Run ingest and processing in separate processes for best performance
    ingest_process = Process(target=ingest_stream)
    process_process = Process(target=process_frame)

    ingest_process.start()
    process_process.start()

    ingest_process.join()
    process_process.join()
