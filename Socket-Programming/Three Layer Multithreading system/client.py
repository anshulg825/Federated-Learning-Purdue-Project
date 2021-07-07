import socket
import time 
import pickle

host = 'localhost'
TCP_PORT = 6000
BUFFER_SIZE = 1024

while True:
    s1 = socket.socket()
    s1.connect((host, TCP_PORT))
    print("connected")
    tm = s1.recv(1024)
    print("The time got from the edge server is %s" % pickle.loads(tm))
    time.sleep(15)
    currentTime = time.ctime(time.time()) + "\r\n"
    s1.sendall(pickle.dumps(currentTime))
    s1.close()