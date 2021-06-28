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
from torchvision import datasets
from torchvision.models.video import r3d_18 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import copy
import transforms as T
import numpy as np 

host = 'localhost' #Enter edge server IP and Port here
TCP_PORT = 6000

data_dir = "./video_data"
annotations = "./test_train_splits"

num_local_update = 3 
batch_size_train = 8
local_client_epoch = 1
lr = 0.001
decay_reg = 0.002
num_frames = 16
clip_steps = 25
learning_rate = 0.001
num_workers = 8
kwargs = {'num_workers':num_workers, 'pin_memory':True} if torch.cuda.is_available() else {'num_workers':num_workers}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
stop_flag = 0

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

train_tfms = torchvision.transforms.Compose([T.ToFloatTensorInZeroOne(),
                                 T.Resize((128, 171)),
                                 T.RandomHorizontalFlip(),
                                 T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                 T.RandomCrop((112, 112))
                               ])  

hmdb51_train = datasets.HMDB51(root = data_dir, annotation_path = annotations, frames_per_clip = num_frames,step_between_clips = clip_steps,train = True,transform = train_tfms, num_workers=num_workers)
total_train_samples = len(hmdb51_train)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        super(DatasetSplit, self).__init__()
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        data = self.dataset[self.idxs[item]]
        return data

def recv(soc, buffer_size=1024):
    received_data = b""
    while str(received_data)[-2] != '.':
        received_data += soc.recv(buffer_size)
    received_data = pickle.loads(received_data)
    return received_data

soc = socket.socket()
soc.connect((host, TCP_PORT))
print("connected")

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

network = VideoRecog_Model()
network = network.to(device)
set_parameter_requires_grad_video(network)
optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=decay_reg)

num_items = int(total_train_samples/4)
#all_idxs = [i for i in range(total_train_samples)]
#idxs = np.random.choice(all_idxs, num_items, replace = False)
idxs = [i for i in range(0,num_items)]   # Replace different values for four clients
train_loader = DataLoader(DatasetSplit(hmdb51_train, idxs), batch_size = batch_size_train, shuffle = True, **kwargs)

def train(num_local_update, train_loader):
    network.train()
    epoch_loss = []
    for epoch in range(num_local_update):  
        batch_loss = []
        for batch_id, data in enumerate(train_loader):
            data, target = data[0].to(device), data[-1].to(device)
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    return network.state_dict(), sum(epoch_loss) / len(epoch_loss)

if (stop_flag != 1):    
    while(True):
        received_data = recv(soc=soc, buffer_size=1024)
        stop_flag = received_data["flag"]

        network_state_dict = received_data["data"]
        network.load_state_dict(network_state_dict)
        updated_state_dict, updated_loss = train(num_epochs=num_local_update,train_loader=train_loader)
        data = {"data":updated_state_dict,"loss":updated_loss,"local_iteration":local_client_epoch}
        soc.sendall(pickle.dumps(data))
        local_client_epoch = local_client_epoch + 1

        if(stop_flag == 1):
            break

soc.close()
print("Socket Closed.\n")