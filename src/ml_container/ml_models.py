import zmq
import json
import time
import msgpack
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind("ipc:///tmp/foosball")
#socket.bind("ipc:///dev/shm/foosball") #use inter process communication and the shared memory folder on tx2. us

while True:
    try:
        print("In while loop")
        data = {"ball_x": 100, "ball_y": 50}
        message = msgpack.packb(data)
        socket.send(message, zmq.DONTWAIT)
    except zmq.ZMQError as e:
        print(f"ZeroMQ error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    time.sleep(1)
