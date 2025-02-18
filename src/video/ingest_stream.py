import cv2

gstreamer = """
gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp, encoding-name=H264, payload=96" ! rtpjitterbuffer latency=50 drop-on-latency=true ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! videoconvert ! autovideosink sync=false
"""

cap = cv2.VideoCapture(gstreamer, cv2.CAP_GSTREAMER)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Input via Gstreamer", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        else:
            break
cap.release()
cv2.destroyAllWindows()