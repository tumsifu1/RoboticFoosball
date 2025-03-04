import cv2
import numpy as np
import torch
from models.binaryClassifier.model import BinaryClassifier
from models.BallLocalization.model_snoutNetBase import BallLocalization
from src.tools import vision_utils
import torch.nn.functional as F
import time
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = BinaryClassifier().to(device)
localizer = BallLocalization().to(device)
COMPUTED_MEAN = [0.1249, 0.1399, 0.1198]
COMPUTED_STD = [0.1205, 0.1251, 0.1123]

preprocess = transforms.Compose([
    transforms.ToTensor(),           # Convert to PyTorch tensor (C, H, W) comment out when testing
    transforms.Normalize(mean=COMPUTED_MEAN, std=COMPUTED_STD)  # Normalize
])

def ingest_stream():
    """Reads frames from the video stream and processes them."""
    gst_str = (
        "udpsrc port=5000 caps=\"application/x-rtp, encoding-name=H264, payload=96\" ! "
        "rtpjitterbuffer latency=50 drop-on-latency=true mode=0 ! "
        "rtph264depay ! h264parse ! nvv4l2decoder enable-max-performance=1 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink"
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

        cv2.imshow("Ball Tracking", frame)
        cv2.waitKey(1)  
        # process_frame(frame)

        next_frame = time.time()
        print(f"Completed frame processing in: {(next_frame - read_frame) * 10 ** 3} ms")

    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame: np.ndarray):
    """Segments the frame into tiles and processes them in batches."""

    begin_proc = time.time()
    tiles = vision_utils.segment_image(frame)
    print(tiles)
    # Normalize and convert tiles to tensors in parallel
    tiles = np.array(tiles, dtype=np.float32)  
    #tiles = torch.tensor(tiles).permute(0, 3, 1, 2).to(device) 
    
    normalized_tiles = preprocess(tiles)

    end_proc = time.time()
    print(f"Pre-processed frame in {(end_proc - begin_proc) * 10 ** 3} ms")
    best_tile_idx = -1

    with torch.no_grad():
        class_start = time.time()
        for tile,i in enumerate(normalized_tiles):
            prediction = classifier(tile).squeeze()  # Shape: (16,)
            
            if prediction == 1:
                best_tile_idx = i
                break
        class_end = time.time()
    
    print(f"Classified in {(class_end - class_start) * 10 ** 3} ms")

    print(predictions)
    start_c2l = time.time()
    # Find the tile with the ball
    best_tile_idx = torch.argmax(predictions).item()
    proc1 = time.time()


    # Extract the best tile and run localizer
    best_tile = tiles[best_tile_idx].unsqueeze(0) 
    proc2 = time.time()

    best_tile = F.interpolate(best_tile, size=(227, 227), mode='bilinear', align_corners=False)

    end_c2l = time.time()

    print(f"Proc 1 {(proc1 - start_c2l) * 10 ** 3} ms, Proc 2 {(proc2 - proc1) * 10 ** 3}, Proc 3 {(end_c2l - proc2) * 10 ** 3}")

    print(f"Model to Model in {(end_c2l - start_c2l) * 10 ** 3} ms")


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
