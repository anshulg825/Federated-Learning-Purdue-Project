import socket
import time
import threading
import pickle

host = 'localhost'
TCP_PORT = 5000
TCP_PORT2 = 6000
BUFFER_SIZE = 1024

cloud_edge_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
edge_client_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cloud_edge_socket1.connect((host, TCP_PORT))
edge_client_socket1.bind((host, TCP_PORT2))                                  
edge_client_socket1.listen(5)                                           
print("Connected")

class SocketThread2(threading.Thread):
    def __init__(self, connection, client_info, buffer_size = 1024):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
    def recv(self):
        received_data = b""
        while str(received_data)[-2] != '.':
            received_data += self.connection.recv(self.buffer_size)
        received_data = pickle.loads(received_data)
        return received_data
    def run(self):
        print("Got a connection from ({})".format(self.client_info))
        while True:
            data = cloud_edge_socket1.recv(1024)
            print("The time got from the cloud is %s", pickle.loads(data))
            currentTime = time.ctime(time.time()) + "\r\n"
            cloud_edge_socket1.sendall(pickle.dumps(currentTime))
            self.connection.sendall(data)
            received_data = self.recv()
            print("The time got from the client is %s",received_data, self.client_info)

for i in range(2):
    connection, client_info = edge_client_socket1.accept()
    socket_thread = SocketThread2(connection=connection,client_info=client_info)
    socket_thread.start()

while True:
    if len(threading.enumerate()) != 1:
        continue
edge_client_socket1.close()

print("Out of loop")    