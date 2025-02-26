import gi
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Initialize GStreamer
Gst.init(None)

# GStreamer pipeline string
pipeline_str = (
    "udpsrc port=5000 caps=application/x-rtp,encoding-name=H264,payload=96 ! "
    "rtpjitterbuffer latency=50 drop-on-latency=true ! "
    "rtph264depay ! h264parse ! nvv4l2decoder ! "
    "nvvidconv ! videoconvert ! appsink name=sink"
)

# Create pipeline
pipeline = Gst.parse_launch(pipeline_str)
appsink = pipeline.get_by_name("sink")

# Set pipeline state to playing
pipeline.set_state(Gst.State.PLAYING)

while True:
    sample = appsink.emit("pull-sample")
    if sample:
        buffer = sample.get_buffer()
        size = buffer.get_size()
        print(f"Received frame of size: {size} bytes")

# Cleanup
pipeline.set_state(Gst.State.NULL)
