import socket
import time
import threading
import pickle
import torch
import torchvision
from torchvision.models.video import r3d_18 
import torch.nn as nn
import os
import copy
import numpy as np 

host = 'localhost' #Enter edge server IP and Port here
TCP_PORT = 5000
TCP_PORT2 = 6000

buffer = []
stop_flag = 0
num_clients = 2
num_edge_agg = 10 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

def recv(soc, buffer_size=1024):
    received_data = b""
    while str(received_data)[-2] != '.':
        received_data += soc.recv(buffer_size)
    received_data = pickle.loads(received_data)
    return received_data

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
        global global_network, buffer, stop_flag
        send_dict = {"data":global_network.state_dict(), "flag": stop_flag}
        send_dict = pickle.dumps(send_dict)
        self.connection.sendall(send_dict)
        received_data = self.recv()
        network_state_dict = received_data["data"]
        buffer.append(network_state_dict)

cloud_edge_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
edge_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print("Socket Created.\n")

edge_client_socket.bind((host, TCP_PORT2))                                  
cloud_edge_socket.connect((host, TCP_PORT))
edge_client_socket.listen(5)                                           
print("Socket is Listening for Connections ....\n")

for i in range(num_clients):
    connection, client_info = edge_client_socket.accept()
    socket_thread = SocketThread(connection=connection,client_info=client_info, buffer_size=4096)
    threads.append(socket_thread)
print("Threading process done!")

while(True):
    
    global global_network, buffer, stop_flag
    received_data = recv(soc=cloud_edge_socket, buffer_size=1024)
    stop_flag = received_data["flag"]
    
    network_state_dict = received_data["data"]
    global_network.load_state_dict(network_state_dict)
    
    for epoch in range(num_edge_agg):
        buffer = []
        for thread in threads:
            thread.start()
            thread.join()

        print("Length of buffer is ", len(buffer))
        average_dict = average_function(receive_buffer = buffer)
        global_network.load_state_dict(average_dict)

    data = {"data": global_network.state_dict()}
    cloud_edge_socket.sendall(pickle.dumps(data))    
    
    if(stop_flag == 1):
        break

cloud_edge_socket.close()
edge_client_socket.close()
print("Socket Closed.\n")