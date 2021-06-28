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
%matplotlib inline

host = 'localhost'                          
port = 5000
BUFFER_SIZE = 1024

num_edges = 2
batch_size = 12
num_frames = 16
clip_steps = 25
bs_test = 64
learning_rate = 0.001
num_comm = 50
global_loss = []
global_correct_accuracy = []
global_epoch_tracker = 0
kwargs = {'num_workers':num_workers, 'pin_memory':True} if torch.cuda.is_available() else {'num_workers':num_workers}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

threads = []
buffer = []

test_tfms =  torchvision.transforms.Compose([
                                             T.ToFloatTensorInZeroOne(),
                                             T.Resize((128, 171)),
                                             T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                             T.CenterCrop((112, 112))
                                             ])

hmdb51_test = torchvision.datasets.HMDB51('video_data/', 'test_train_splits/', num_frames,
                                                step_between_clips = clip_steps, fold=1, train=False,
                                                transform=test_tfms, num_workers=num_workers)

print(f"number of test samples {len(hmdb51_test)}")
test_loader  = DataLoader(hmdb51_test, batch_size=bs_test, shuffle=False, **kwargs)


class VideoRecog_Model(nn.Module):
  def __init__(self):
      super(VideoRecog_Model, self).__init__()
      self.base_model = nn.Sequential(*list(r3d_18(pretrained=True).children())[:-1])
      self.fc = nn.Linear(512, 51)
  def forward(self, x):
      out = self.base_model(x).squeeze(4).squeeze(3).squeeze(2)
      out = torch.log_softmax(self.fc(out), dim=1)
      return out

def set_parameter_requires_grad_video(model):
    for param in model.parameters():
        param.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

global_network = VideoRecog_Model().to(device)
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
    return total_loss, correct

def average_function(receive_buffer):
    w_avg = copy.deepcopy(receive_buffer[0])
    for key in w_avg.keys():
        w_avg[key] = (receive_buffer[0][key] + receive_buffer[1][key])/2.
    return w_avg

class SocketThread(threading.Thread):
    
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
        global global_network, buffer, global_epoch_tracker
        send_dict = {"data":global_network.state_dict(), "flag":0}
        if global_epoch_tracker == (num_comm-1):
            send_dict["flag"] = 1
        send_dict = pickle.dumps(send_dict)
        self.connection.sendall(send_dict)
        received_data = self.recv()
        network_state_dict = received_data["data"]
        buffer.append(network_state_dict)
            
serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
print("Socket Created.\n")

serversocket.bind((host, port))  
print("Socket Bound to IPv4 Address & Port Number.\n")                                

serversocket.listen(5)            
print("Socket is Listening for Connections ....\n")

for i in range(num_edges):
    connection, client_info = soc.accept()
    socket_thread = SocketThread(connection=connection,client_info=client_info, buffer_size=4096)
    threads.append(socket_thread)
print("Threading process done!")

for epoch in range(num_comm):
    global global_network, buffer, global_epoch_tracker
    buffer = []
    for thread in threads:
        thread.start()
        thread.join()
    print(len(buffer))
    average_dict = average_function(receive_buffer = buffer)
    global_network.load_state_dict(average_dict)
    loss, accuracy = validation(model = global_network)
    print("Epoch {} has {} loss and {} accuracy".format(epoch, loss, accuracy))
    global_correct_accuracy.append([accuracy,epoch])
    global_loss.append([loss,epoch])
    global_epoch_tracker += 1

x = []
y = []
for i in global_correct_accuracy:
    x.append(i[1])
    y.append(i[0])
plt.plot(x,y)
plt.show()

serversocket.close()
print("Socket Closed.\n")