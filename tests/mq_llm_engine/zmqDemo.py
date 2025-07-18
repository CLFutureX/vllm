import zmq
import time

context = zmq.Context()
socket = context.socket(zmq.REP)

socket.bind('tcp://*:5555')

while True:
    message = socket.recv_string()
    print(f"收到请求 : {message}")

    time.sleep(1)

    socket.send_string(f"对 {message} 的响应")