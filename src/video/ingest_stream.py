import gi
import cv2
import numpy as np
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import matplotlib.pyplot as plt
import queue
import threading 
Gst.init(None)


frame_queue = queue.Queue(maxsize=10)

def frame_display():
    """Continuously fetch frames from the queue and display them."""
    cv2.namedWindow("Stream", cv2.WINDOW_NORMAL)
    while True:
        frame = frame_queue.get()
        if frame is None:
            break  # Stop the thread
        cv2.imshow("Stream", frame)
        cv2.waitKey(1)

display_thread = threading.Thread(target=frame_display, daemon=True)
display_thread.start()


def on_new_sample(sink):
    """Callback function for retrieving frames"""
    sample = sink.emit("pull-sample")
    if sample:
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_int("width")[1]
        height = caps.get_structure(0).get_int("height")[1]
        success, map_info = buffer.map(Gst.MapFlags.READ)
        print(width, height, map_info)
        if success:
            print(f"Buffer size: {len(map_info.data)} | Expected: {width * height * 3}")
            
            if len(map_info.data) == width * height * 3:
                frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3))
                print("Frame successfully reshaped.")
                frame = frame[:, :, ::-1]
                try:
                    frame_queue.put_nowait(frame)  # Send frame to display thread
                except:
                    print("Queue full")  
            else:
                print("Skipping frame due to size mismatch.")

            buffer.unmap(map_info)
    
    return Gst.FlowReturn.OK

pipeline = Gst.parse_launch(
    "udpsrc port=5000 caps=\"application/x-rtp, encoding-name=H264, payload=96\" ! "
    "rtpjitterbuffer latency=50 drop-on-latency=true ! rtph264depay ! "
    "h264parse ! nvv4l2decoder ! nvvidconv ! videoconvert ! "
    "video/x-raw, format=RGB ! appsink name=sink emit-signals=True max-buffers=1 drop=True"
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
    cv2.destroyAllWindows()
