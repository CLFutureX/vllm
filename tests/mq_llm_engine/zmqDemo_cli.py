import zmq

context = zmq.Context()
socket = context.socket(zmq.REQ)

socket.connect("tcp://localhost:5555")

for request_num in range(5):
    request = f"请求#{request_num}"
    print(f"client 发送请求: {request}")

    socket.send_string(request)

    response = socket.recv_string()
    print(f"client 收到响应： {response}")