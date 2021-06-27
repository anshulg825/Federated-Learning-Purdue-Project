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

num_classes = 51
batch_size = 12
log_interval = 1
# num_epochs = 200
num_frames = 8
clip_steps = 50

val_split = 0.1

random_seed = 1
learning_rate = 0.001
momentum = 0.9

client_num_epochs = 2  
num_iter = 10
num_users = 3
global_epoch_tracker = 0
global_loss = []
global_correct_accuracy = []
client_loss = []


torch.manual_seed(random_seed)
np.random.seed(random_seed)


class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      self.base_model = nn.Sequential(*list(models.video.r3d_18(pretrained=True).children())[:-1])
      self.fc = nn.Linear(512, 51)

  def forward(self, x):
      out = self.base_model(x)
      out = out.flatten(1)
      out = self.fc(out)
      return out

def set_parameter_requires_grad_video(model):
    for param in model.parameters():
        param.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

global_network = Net()
global_network = global_network.to("cuda")
set_parameter_requires_grad_video(global_network)
criterion = nn.CrossEntropyLoss()

def validation(model):
    global test_loader, criterion
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            
            test_loss += criterion(output, target).item()
            prediction = output.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    test_loss /= len(test_loader)
    correct /= len(test_loader.dataset)

    return (test_loss, correct)


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