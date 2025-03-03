import cv2
import numpy as np
import torch
import concurrent.futures
from models.binaryClassifier.model import BinaryClassifier
from models.BallLocalization.model_snoutNetBase import BallLocalization
from src.tools import vision_utils
import torch.nn.functional as F
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = BinaryClassifier().to(device)
localizer = BallLocalization().to(device)

def ingest_stream():
    """Reads frames from the video stream and processes them."""
    gst_str = (
        "udpsrc port=5000 caps=\"application/x-rtp, encoding-name=H264, payload=96\" ! "
        "rtpjitterbuffer latency=50 drop-on-latency=true ! rtph264depay ! h264parse ! "
        "nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink"
    )

    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Failed to open video")
        return

    while True:
        read_frame = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        process_frame(frame)

        next_frame = time.time()
        print(f"Completed frame processing in: {(next_frame - read_frame) * 10 ** 3} ms")

    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame: np.ndarray):
    """Segments the frame into tiles and processes them in batches."""

    begin_proc = time.time()
    tiles = vision_utils.segment_image(frame)

    # Normalize and convert tiles to tensors in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        tiles = list(executor.map(lambda tile: torch.tensor(tile.astype(np.float32) / 255.0).permute(2, 0, 1), tiles))

    # Minibatch the tiles
    tile_batch = torch.stack(tiles).to(device) 

    end_proc = time.time()
    print(f"Pre-processed frame in {(end_proc - begin_proc) * 10 ** 3} ms")
    
    with torch.no_grad():
        class_start = time.time()
        predictions = classifier(tile_batch).squeeze()  # Shape: (16,)
        class_end = time.time()
    
    print(f"Classified in {(class_end - class_start) * 10 ** 3} ms")
    # Find the tile with the **highest confidence score**
    best_tile_idx = torch.argmax(predictions).item()

    # Extract the best tile and run localizer
    best_tile = tile_batch[best_tile_idx].unsqueeze(0)  # Shape: (1, C, H, W)

    best_tile = F.interpolate(best_tile, size=(227, 227), mode='bilinear', align_corners=False)

    with torch.no_grad():
        local_start = time.time()
        local_x, local_y = localizer(best_tile).cpu().numpy().flatten()
        local_end = time.time()

    print(f"Localized in {(local_end - local_start) * 10 ** 3} ms")

    # Convert to absolute coordinates
    row, col = divmod(best_tile_idx, 8)
    abs_x, abs_y = vision_utils.rebuild_absolute_coordinates(row, col, local_x, local_y, 1280 // 8, 720 // 8)

    print(f"Ball detected at: ({abs_x}, {abs_y})")

    cv2.circle(frame, (int(abs_x), int(abs_y)), radius=5, color=(0, 0, 255), thickness=-1)  # Red dot
    cv2.imshow("Ball Tracking", frame)
    cv2.waitKey(1)  

def main():
    """Main function to load models and start video processing."""
    classifier.load_state_dict(torch.load('./src/weights/classifier.pth', map_location=device))
    localizer.load_state_dict(torch.load('./src/weights/localizer.pth', map_location=device))
    classifier.eval()
    localizer.eval()

    ingest_stream()

if __name__ == "__main__":
    main()
