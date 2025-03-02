import cv2
import numpy as np

def receiveStream():
    gst_str = (
        "udpsrc port=5000 caps=\"application/x-rtp, encoding-name=H264, payload=96\" ! "
        "rtpjitterbuffer latency=50 drop-on-latency=true ! rtph264depay ! h264parse ! "
        "nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink"
    )

    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Failed to open video")
        return

    # Indefinitely grab frames from the stream and read into an 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        cv2.imshow("Jetson Stream", frame)
        cv2.waitKey(1) # Add a msec wait to allow for output



    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    receiveStream()
