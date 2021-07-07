import socket
import time
import threading
import pickle

def recv(soc, buffer_size=1024):
    received_data = b""
    while str(received_data)[-2] != '.':
        received_data += soc.recv(buffer_size)
    received_data = pickle.loads(received_data)
    return received_data

class SocketThread2(threading.Thread):
    def __init__(self, connection, client_info, buffer_size = 1024):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        print("Got a connection from ({})".format(self.client_info))
    def recv(self):
        received_data = b""
        while str(received_data)[-2] != '.':
            received_data += self.connection.recv(self.buffer_size)
        received_data = pickle.loads(received_data)
        return received_data
    def run(self):
        currentTime = time.ctime(time.time()) + "\r\n"
        self.connection.sendall(pickle.dumps(currentTime))
        received_data = self.recv()
        print("The time got from the client is %s",received_data, self.client_info)

host = 'localhost'
TCP_PORT = 5000
TCP_PORT2 = 6000
BUFFER_SIZE = 1024

edge_client_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
edge_client_socket2.bind((host, TCP_PORT2))                                  
edge_client_socket2.listen(5)                                           
print("Connected")

while True:
    cloud_edge_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cloud_edge_socket.connect((host, TCP_PORT))
    received_data = recv(soc=cloud_edge_socket, buffer_size=1024)
    print(received_data)
    threads = []
    for i in range(2):
        connection, client_info = edge_client_socket2.accept()
        socket_thread = SocketThread2(connection=connection,client_info=client_info)
        threads.append(socket_thread)
    for thread in threads:
        thread.start()
        thread.join()      
    currentTime = time.ctime(time.time()) + "\r\n"
    cloud_edge_socket.sendall(pickle.dumps(currentTime))
    cloud_edge_socket.close()
    print("Done")