"""
Run these before starting code
! apt-get install libavformat-dev libavdevice-dev
! pip install av==6.2.0
! wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
! wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar
! wget https://raw.githubusercontent.com/pytorch/vision/6de158c473b83cf43344a0651d7c01128c7850e6/references/video_classification/transforms.py
"""

import socket
import time
import threading
import pickle

import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.models.video import r3d_18 

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import os
import copy
import transforms as T

import matplotlib.pyplot as plt
import numpy as np 

host = 'localhost'                          
port = 5000
BUFFER_SIZE = 1024


batch_size = 12
num_frames = 16
clip_steps = 25
random_seed = 1
learning_rate = 0.001
client_num_epochs = 2  
num_iter = 10
global_epoch_tracker = 0
global_loss = []
global_correct_accuracy = []
client_loss = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(SEED)

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

def validation(model):
    global test_loader
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            data, target = data[0].to(device), data[-1].to(device)
            output = model(data)
            total_loss += (F.nll_loss(output, target)).item()
            pred = output.argmax(dim=1, keepdim=True)  
            correct += pred.eq(target.view_as(pred)).sum().item()
    total_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)
    return (total_loss, correct)

class SocketThread(threading.Thread):

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
            
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
serversocket.bind((host, port))                                  
serversocket.listen(5)            
print("Waiting for connections")

for i in range(2):
    connection, client_info = serversocket.accept()
    socket_thread = SocketThread(connection=connection,client_info=client_info)
    socket_thread.start()

while True:
    if len(threading.enumerate()) != 1:
        continue
serversocket.close()
    
print("Out of loop")