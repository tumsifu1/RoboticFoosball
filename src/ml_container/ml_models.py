import zmq
import json
import time
import msgpack

context = zmq.Context()  # todo sudo apt docker and make sure you have daemon running (at least on mac it didn't start with it)
socket = context.socket(zmq.PUB)  # Changed from PAIR to PUB
socket.bind("tcp://0.0.0.0:5555")

print("ML Container is running... Waiting for connection.")
time.sleep(2)  # give the motor container time to start

while True:  # todo add process for ingestion and running the models here
    try:
        print("In while loop")
        data = {"ball_x": 100, "ball_y": 50}
        message = msgpack.packb(data)
        socket.send(message)  # PUB does not need zmq.DONTWAIT
    except zmq.ZMQError as e:
        print(f"ZeroMQ error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    time.sleep(1)  # todo remove this sleep it's just for debugging so you don't get lots of messages
