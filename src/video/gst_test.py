from threading import Thread
from time import sleep
import sys
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")

from gi.repository import Gst, GstApp, GLib

# Placeholder to prevent ignores of the import
_ = GstApp

# Initialize GStreamer
Gst.init(None)

main_loop = GLib.MainLoop()

# Generate thread to run the main loop
ml_thread = Thread(target=main_loop.run)
ml_thread.start()

# Change `autovideosrc` to `v4l2src` for Linux or keep for macOS compatibility
pipeline = Gst.parse_launch("autovideosrc ! decodebin ! videoconvert ! osxvideosink")

# Set the pipeline to the PLAYING state
pipeline.set_state(Gst.State.PLAYING)

try:
    while True:
        sleep(0.1)  # Main thread sleeps to avoid exiting
except KeyboardInterrupt:
    print("Stopping pipeline...")

# Clean up on exit
pipeline.set_state(Gst.State.NULL)
main_loop.quit()
ml_thread.join()  # Ensure the main loop thread stops gracefully
print("Pipeline stopped.")
