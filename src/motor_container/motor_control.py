import zmq
import msgpack 


context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.connect("ipc:///tmp/foosball")
#socket.bind("ipc:///dev/shm/foosball") #use inter process communication and the shared memory folder on tx2. us


while True:
    
    message = socket.recv()  # Receive binary message
    data = msgpack.unpackb(message)  # Convert back to dictionary

    ball_x, ball_y = data["ball_x"], data["ball_y"]
    print(f"Received Ball Position: X={ball_x}, Y={ball_y}")