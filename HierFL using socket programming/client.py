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

data_dir = "./video_data"
annotations_data = "./test_train_splits"

#B, E sent from central server
# n_epochs = 3 
# batch_size_train = 64
# batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
decay_reg = 0.02
# log_interval = 10
num_classes = 51
num_frames = 8
clip_steps = 50
learning_rate = 0.001
momentum = 0.9
val_split = 0.1

random_seed = 1
torch.manual_seed(random_seed)
np.random.seed(random_seed)

data_transforms = {
    'train': transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((128, 171)),
        T.RandomHorizontalFlip(),
        T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        T.RandomCrop((112, 112))    
    ]),
    'test': transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((128, 171)),
        T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        T.CenterCrop((112, 112))
    ]),
}

hmdb51_train = datasets.HMDB51(root = data_dir, annotation_path = annotations_data, frames_per_clip = num_frames,step_between_clips = clip_steps,train = True,transform = data_transforms['train'])

total_train_samples = len(hmdb51_train)
total_val_samples = round(val_split * total_train_samples)

hmdb51_test = datasets.HMDB51(root = data_dir, annotation_path = annotations_data, frames_per_clip = num_frames,step_between_clips = clip_steps,train = False,transform = data_transforms['test'])

class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def recv(soc, buffer_size=1024):
    received_data = b""
    while str(received_data)[-2] != '.':
        received_data += soc.recv(buffer_size)

    received_data = pickle.loads(received_data)

    return received_data

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

network = Net()
network = network.to("cuda")
set_parameter_requires_grad_video(network)
criterion = nn.CrossEntropyLoss(reduction="mean")
optimizer = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum,weight_decay=decay_reg)

def train(num_epochs,train_loader):
    network.train()
    epoch_loss = []
    for iter in range(num_epochs):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to("cuda"), labels.to("cuda")
            optimizer.zero_grad()
            output = network(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    return network.state_dict(), sum(epoch_loss) / len(epoch_loss)


host = 'localhost'
TCP_PORT = 6000
BUFFER_SIZE = 1024

s1 = socket.socket()
s1.connect((host, TCP_PORT))
print("connected")

while True:
    tm = s1.recv(1024)
    print("The time got from the edge server is %s" % pickle.loads(tm))
    currentTime = time.ctime(time.time()) + "\r\n"
    s1.sendall(pickle.dumps(currentTime))