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