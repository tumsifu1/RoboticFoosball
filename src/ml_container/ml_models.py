import zmq
import time
from ingest_stream import process_frame

# ZeroMQ Context
context = zmq.Context()

# Create PUSH socket
socket = context.socket(zmq.PUSH)
socket.bind("ipc:///tmp/ball_updates")  # IPC for low latency
time.sleep(2)
ingest_stream()

def ingest_stream():
    Gst.init(None)
    """Reads frames from the video stream and processes them using NVIDIA decoder (without CUDA)."""
    def on_new_sample(sink):
        """Callback function for retrieving frames"""
        sample = sink.emit("pull-sample")
        if sample:
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            width = caps.get_structure(0).get_int("width")[1]
            height = caps.get_structure(0).get_int("height")[1]
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                if len(map_info.data) == width * height * 3:
                    frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape((height, width, 3))
                    coords = process_frame(frame)
                    socket.send_string(coords)

                buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK

    pipeline = Gst.parse_launch(
        "udpsrc port=5000 buffer-size=200000 caps=\"application/x-rtp, encoding-name=H264, payload=96\" ! "
        "rtpjitterbuffer latency=50 drop-on-latency=true do-lost=true ! rtph264depay ! "
        "h264parse ! queue max-size-buffers=3 leaky=downstream ! nvv4l2decoder enable-max-performance=1 disable-dpb=true "
        " ! nvvidconv ! videoconvert ! video/x-raw, format=RGB ! appsink name=sink emit-signals=True max-buffers=1 drop=True"
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