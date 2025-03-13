import zmq
import time

# ZeroMQ Context
context = zmq.Context()

# Create PULL socket
socket = context.socket(zmq.PULL)
socket.setsockopt(zmq.RCVHWM, 1) 
socket.setsockopt(zmq.IMMEDIATE, 1)

# Try connecting with retry
connected = False
while not connected:
    try:
        socket.connect("ipc:///tmp/ball_updates")
        connected = True
        print("Successfully connected to ml_container")
    except zmq.error.ZMQError as e:
        print(f"Connection failed: {e}, retrying in 2 seconds...")
        time.sleep(2)


while True:
    try:
        # Poll with a very short timeout (1ms)
        if socket.poll(1):
            message = socket.recv_string(zmq.NOBLOCK)
            print(f"Received ball coordinates: {message}")
            # Process message
        else:
            # No message available
            time.sleep(0.001)  # Small sleep to prevent CPU spinning
    except zmq.ZMQError:
        pass  # Handle errors
