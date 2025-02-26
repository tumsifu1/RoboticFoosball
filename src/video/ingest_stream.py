import cv2
import numpy as np

# Define the GStreamer pipeline
gst_pipeline = (
    "udpsrc port=5000 caps=application/x-rtp, encoding-name=H264, payload=96 ! "
    "rtpjitterbuffer latency=50 drop-on-latency=true ! "
    "rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=true"
)

# Open the GStreamer pipeline with OpenCV
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Couldn't open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Couldn't receive frame (stream may have stopped).")
        break

    # Display the frame
    cv2.imshow("Jetson TX2 Video Stream", frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
