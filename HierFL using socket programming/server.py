import socket
import time
import threading
import pickle

import torch
import torchvision
from torchvision import datasets, models, transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import copy
import transforms as T

import matplotlib.pyplot as plt
import numpy as np 

class SocketThread1(threading.Thread):
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
            currentTime = time.ctime(time.time()) + "\r\n"
            self.connection.sendall(pickle.dumps(currentTime))
            received_data = self.recv()
            print("The time got from the edge server is %s",received_data, self.client_info)
            
host = 'localhost'                          
port = 5000
BUFFER_SIZE = 1024

serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
serversocket.bind((host, port))                                  
serversocket.listen(5)            
print("Waiting for connections")

for i in range(2):
    connection, client_info = serversocket.accept()
    socket_thread = SocketThread1(connection=connection,client_info=client_info)
    socket_thread.start()

while True:
    if len(threading.enumerate()) != 1:
        continue
serversocket.close()
    
print("Out of loop")