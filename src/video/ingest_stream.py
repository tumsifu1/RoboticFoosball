import cv2
import numpy as np
import time
import sys

class SoftwareDecoderReceiver:
    def __init__(self, port=5000):
        self.port = port
        
        # Define the GStreamer pipeline with software decoder
        self.gst_pipeline = (
            f"udpsrc port={port} caps=\"application/x-rtp, encoding-name=H264, payload=96\" ! "
            "rtpjitterbuffer latency=50 drop-on-latency=true ! "
            "rtph264depay ! h264parse ! "
            "avdec_h264 ! "  # Software decoder instead of nvv4l2decoder
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink max-buffers=2 drop=true sync=false"
        )
        
        print(f"Using pipeline: {self.gst_pipeline}")
        
        # Create the capture object
        self.cap = cv2.VideoCapture(self.gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print("Failed to open video capture")
            sys.exit(1)
        
        print("Video capture opened successfully")
    
    def run(self):
        try:
            while True:
                # Read a frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to get frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Display the frame
                cv2.imshow('Video Stream', frame)
                
                # Here you would pass the frame to your ML pipeline
                # self.process_with_ml(frame)
                
                # Check for ESC key
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    break
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("Capture released")
    
    def process_with_ml(self, frame):
        # Placeholder for your ML processing
        # This is where you would integrate your machine learning pipeline
        pass

if __name__ == "__main__":
    receiver = SoftwareDecoderReceiver(port=5000)
    receiver.run()
