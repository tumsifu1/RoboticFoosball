import cv2
import time

def ingest_stream():
    """Reads frames from the video stream and processes them."""
    gst_str = (
        "udpsrc port=5000 caps=\"application/x-rtp, encoding-name=H264, payload=96\" ! "
        "rtpjitterbuffer latency=50 drop-on-latency=true ! "
        "rtph264depay ! h264parse ! nvv4l2decoder ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink sync=false"
    )

    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Failed to open video stream")
        return

    while True:
        read_frame = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to read frame")
            continue  # Instead of breaking, we retry to avoid dropping the stream

        cv2.imshow("Ball Tracking", frame)
        cv2.waitKey(1)  

        next_frame = time.time()
        print(f"Frame processing time: {(next_frame - read_frame) * 1000:.2f} ms")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ingest_stream()
