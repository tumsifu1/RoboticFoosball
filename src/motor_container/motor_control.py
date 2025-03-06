import zmq
import msgpack

context = zmq.Context()
socket = context.socket(zmq.SUB) 
socket.connect("tcp://publisher:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")
socket.setsockopt(zmq.CONFLATE, 1)
print("Motor Container is running... Waiting for data.")

while True:
    message = socket.recv()  # Blocking call, waits for data
    data = msgpack.unpackb(message)
    print(f"Received Ball Position: X={data['ball_x']}, Y={data['ball_y']}")
