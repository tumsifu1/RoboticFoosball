import zmq

# ZeroMQ Context
context = zmq.Context()

# Create PULL socket
socket = context.socket(zmq.PULL)
socket.connect("ipc:///tmp/ball_updates")  # Connect to IPC channel

while True:
    # Receive ball coordinates
    message = socket.recv_string()
    print(f"Received ball coordinates: {message}")
    
    # Parse (x, y) coordinates
    x, y = map(int, message.split(","))
    
    # TODO: Add motor movement logic here
    # move_motors(x, y)
