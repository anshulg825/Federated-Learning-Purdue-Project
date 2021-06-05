from models.cifar_cnn_3conv_layer import cifar_cnn_3conv, cifar_cnn_3conv_specific, cifar_cnn_3conv_shared
from models.cifar_resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from models.mnist_cnn import mnist_lenet
from models.mnist_logistic import LogisticRegression
import torch.optim as optim
import torch.nn as nn

class MTL_Model(object):
    def __init__(self, shared_layers, specific_layers, learning_rate, lr_decay, lr_decay_epoch, momentum, weight_decay):
        self.shared_layers = shared_layers
        self.specific_layers = specific_layers
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.momentum = momentum
        self.weight_decay = weight_decay
    #   construct the parameter
        param_dict = [{"params": self.shared_layers.parameters()}]
        if self.specific_layers:
            param_dict += [{"params": self.specific_layers.parameters()}]
        self.optimizer = optim.SGD(params = param_dict,
                                  lr = learning_rate,
                                  momentum = momentum,
                                  weight_decay=weight_decay)
        self.optimizer_state_dict = self.optimizer.state_dict()
        self.criterion = nn.CrossEntropyLoss()
    def exp_lr_sheduler(self, epoch):
        if  (epoch + 1) % self.lr_decay_epoch:
            return None
        for param_group in self.optimizer.param_groups:
            # print(f'epoch{epoch}')
            param_group['lr'] *= self.lr_decay
            return None
    def step_lr_scheduler(self, epoch):
        if epoch < 150:
            for param_group in self.optimizer.param_groups:
                # print(f'epoch{epoch}')
                param_group['lr'] = 0.1
        elif epoch >= 150 and epoch < 250:
            for param_group in self.optimizer.param_groups:
                # print(f'epoch{epoch}')
                param_group['lr'] = 0.01
        elif epoch >= 250:
            for param_group in self.optimizer.param_groups:
                # print(f'epoch{epoch}')
                param_group['lr'] = 0.001
    def print_current_lr(self):
        for param_group in self.optimizer.param_groups:
            print(param_group['lr'])
    def optimize_model(self, input_batch, label_batch):
        self.shared_layers.train(True)
        if self.specific_layers:
            self.specific_layers.train(True)
        if self.specific_layers:
            output_batch = self.specific_layers(self.shared_layers(input_batch))
        else:
            output_batch = self.shared_layers(input_batch)
        self.optimizer.zero_grad()
        batch_loss = self.criterion(output_batch, label_batch)
        batch_loss.backward()
        self.optimizer.step()
        # self.optimizer_state_dict = self.optimizer.state_dict()
        return batch_loss.item()
    def test_model(self, input_batch):
        self.shared_layers.train(False)
        with torch.no_grad():
            if self.specific_layers:
                output_batch = self.specific_layers(self.shared_layers(input_batch))
            else:
                output_batch = self.shared_layers(input_batch)
        self.shared_layers.train(True)
        return output_batch
    def update_model(self, new_shared_layers):
        self.shared_layers.load_state_dict(new_shared_layers)

def initialize_model(args, device):
    if args.mtl_model:
        print('Using different task specific layer for each user')
        if args.dataset == 'cifar10':
            if args.model == 'cnn_complex':
                shared_layers = cifar_cnn_3conv_shared(input_channels=3)
                feature_out_dim = shared_layers.feature_out_dim()
                specific_layers = cifar_cnn_3conv_specific(input_channels=feature_out_dim,
                                                           output_channels=10)
            else:
                raise ValueError('Model not implemented for CIFAR-10')
        else:
            raise ValueError('The dataset is not implemented for mtl yet')
        if args.cuda:
            shared_layers = shared_layers.cuda(device)
            specific_layers = specific_layers.cuda(device)
    elif args.global_model:
        print('Using same global model for all users')
        if args.dataset == 'cifar10':
            if args.model == 'cnn_complex':
                shared_layers = cifar_cnn_3conv(input_channels=3, output_channels=10)
                specific_layers = None
            elif args.model == 'resnet18':
                shared_layers = ResNet18()
                specific_layers = None
            else:
                raise ValueError('Model not implemented for CIFAR-10')
        elif args.dataset == 'mnist':
            if args.model == 'lenet':
               shared_layers = mnist_lenet(input_channels=1, output_channels=10)
               specific_layers = None
            elif args.model == 'logistic':
               shared_layers = LogisticRegression(input_dim=1, output_dim=10)
               specific_layers = None
            else:
                raise ValueError('Model not implemented for MNIST')
        else:
            raise ValueError('The dataset is not implemented for mtl yet')
        if args.cuda:
            shared_layers = shared_layers.cuda(device)
    else: raise ValueError('Wrong input for the --mtl_model and --global_model, only one is valid')
    model = MTL_Model(shared_layers = shared_layers,
                      specific_layers = specific_layers,
                      learning_rate= args.lr,
                      lr_decay= args.lr_decay,
                      lr_decay_epoch= args.lr_decay_epoch,
                      momentum= args.momentum,
                      weight_decay = args.weight_decay)
    return model