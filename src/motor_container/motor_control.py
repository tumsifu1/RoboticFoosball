import zmq
import time
from trajectory import get_velocities
from game_init import game_init

context = zmq.Context()

# Create SUB socket
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.RCVHWM, 1)
socket.setsockopt(zmq.IMMEDIATE, 1)
socket.connect("tcp://localhost:5555")  # Connect to container's IP
socket.setsockopt_string(zmq.SUBSCRIBE, "BALL ")


game_init()

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

            if prev:
                 get_velocities(prev, curr)

            prev = curr
            
    except zmq.ZMQError as e:
        print(f"Error receiving: {e}")

