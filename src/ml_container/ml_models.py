import zmq
import time

# ZeroMQ Context
context = zmq.Context()

# Create PUSH socket
socket = context.socket(zmq.PUSH)
socket.bind("ipc:///tmp/ball_updates")  # IPC for low latency
time.sleep(2)
while True:
    # Replace with actual ball position from ML model
    ball_coordinates = "100,200"  # Example coordinates
    socket.send_string(ball_coordinates)
    print(f"Sent ball coordinates: {ball_coordinates}")
    
    time.sleep(0.01)  # Adjust based on frame rate (e.g., 100 FPS → 10ms)
