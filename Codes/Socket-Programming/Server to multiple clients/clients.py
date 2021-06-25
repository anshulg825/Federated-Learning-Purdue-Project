import socket
import time 
import pickle

host = 'localhost'
TCP_PORT = 5000
BUFFER_SIZE = 1024

s1 = socket.socket()
s1.connect((host, TCP_PORT))
print("connected")

while True:
    tm = s1.recv(1024)
    print("The time got from the server is %s" % pickle.loads(tm))
    currentTime = time.ctime(time.time()) + "\r\n"
    s1.sendall(pickle.dumps(currentTime))