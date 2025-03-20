import zmq
import time
from trajectory import get_velocities

# ZeroMQ Context
context = zmq.Context()

# Create PULL socket
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.RCVHWM, 1) 
socket.setsockopt(zmq.IMMEDIATE, 1)
socket.connect("ipc:///tmp/ball_updates")
socket.setsockopt_string(zmq.SUBSCRIBE, "BALL ")
print(f"ZMQ PUB/SUB setup complete - subscribed to topic: BALL")

handshake_socket = context.socket(zmq.REQ)
handshake_socket.connect("ipc:///tmp/ball_handshake")
handshake_socket.send_string("READY")
handshake_socket.recv_string()  # Wait for acknowledgment
print("Connected and synchronized with ml_container")

counter = 0

curr = None
prev = None

while True:
    try:
        if socket.poll(1):  # Poll with 1ms timeout
            message = socket.recv_string(zmq.NOBLOCK)[5:]
            print(f"Received ball coordinates: {message}, {time.time()}")
            
            curr = tuple(map(int, message.split(",")))

            print("prev: ", prev, "     curr: ", curr)

            if curr:
                 get_velocities(prev, curr)

            prev = curr
            
    except zmq.ZMQError as e:
        print(f"Error receiving: {e}")

