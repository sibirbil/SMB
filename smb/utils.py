#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import math
import matplotlib.pyplot as plt 

# In order to use GPU
use_GPU = torch.cuda.is_available()

##############################################################################
# Function definitions (taken from https://github.com/IssamLaradji/sls/blob/master/sls/utils.py)
##############################################################################



def get_grad_list(params):
    return [p.grad for p in params]

def compute_grad_norm(grad_list):
    grad_norm = 0.
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm


##############################################################################
### METRICS (taken from https://github.com/IssamLaradji/sls/blob/master/src/metrics.py)
##############################################################################


def softmax_loss(model, images, labels, backwards=False):
    logits = model(images)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(logits, labels.view(-1))

    if backwards and loss.requires_grad:
        loss.backward()

    return loss

def softmax_accuracy(model, images, labels):
    logits = model(images)
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()

    return acc

def compute_loss(model, dataset):
    metric_function = softmax_loss
    
    model.eval()

    loader = DataLoader(dataset, drop_last=False, batch_size=1024)

    score_sum = 0.
    for images, labels in loader:
        if use_GPU:
            images, labels = images.cuda(), labels.cuda()

        score_sum += metric_function(model, images, labels).item() * images.shape[0] 
            
    score = float(score_sum / len(loader.dataset))

    return score

def compute_accuracy(model, dataset):
    metric_function = softmax_accuracy
    
    model.eval()

    loader = DataLoader(dataset, drop_last=False, batch_size=1024)

    score_sum = 0.
    for images, labels in loader:
        if use_GPU:
            images, labels = images.cuda(), labels.cuda()

        score_sum += metric_function(model, images, labels).item() * images.shape[0] 
            
    score = float(score_sum / len(loader.dataset))

    return score

##############################################################################
### DATA #####################################################################
##############################################################################

def get_dataset(dataset_name, batch_size):
    
    if dataset_name == "mnist":

        train_set = torchvision.datasets.MNIST("Datasets", train=True,
                                       download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.5,), (0.5,))
                                       ]))
        
        test_set = torchvision.datasets.MNIST("Datasets", train=False,
                                       download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.5,), (0.5,))
                                       ]))
        
        train_loader = torch.utils.data.DataLoader(train_set,
                                      drop_last=True,
                                      shuffle=True,
                                      batch_size=batch_size)
        
        return train_set, test_set, train_loader
    
    
    if dataset_name == "cifar10":
        
        transform_function = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),
                                         torchvision.transforms.RandomHorizontalFlip(),
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ])

        train_set = torchvision.datasets.CIFAR10(root='Datasets',
                                         train=True,
                                         download=True,
                                         transform=transform_function
                                        )

        test_set = torchvision.datasets.CIFAR10(root="Datasets", 
                                        train=False,
                                        download=True,
                                        transform=transform_function
                                       )

        train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=batch_size,
                                           drop_last=True,
                                           shuffle=True,
                                          )
        
        return train_set, test_set, train_loader
    
    
    if dataset_name == "cifar100":
    
        transform_function = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),
                                         torchvision.transforms.RandomHorizontalFlip(),
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
                                        ])

        train_set = torchvision.datasets.CIFAR100(
            root='Datasets',
            train=True,
            download=True,
            transform=transform_function)

        test_set = torchvision.datasets.CIFAR100(
            root='Datasets',
            train=False,
            download=True,
            transform=transform_function)

        train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=batch_size,
                                           drop_last=True,
                                           shuffle=True,
                                          )
        
        return train_set, test_set, train_loader
    
##############################################################################
### MODELS (taken from https://github.com/IssamLaradji/sls/blob/master/src/models.py)
##############################################################################



class Mlp(nn.Module):
    def __init__(self, input_size=784,
                 hidden_sizes=[512, 256],
                 n_classes=10,
                 bias=True, dropout=False):
        super().__init__()

        self.dropout=dropout
        self.input_size = input_size
        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size, bias=bias) for
                                            in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
        self.output_layer = nn.Linear(hidden_sizes[-1], n_classes, bias=bias)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            Z = layer(out)
            out = F.relu(Z)

            if self.dropout:
                out = F.dropout(out, p=0.5)

        logits = self.output_layer(out)

        return logits
    

class ResNet(nn.Module):
    
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        block = BasicBlock
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super().__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
class Bottleneck_DenseNet(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out
    
class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

def DenseNet121(num_classes):
    return DenseNet(Bottleneck_DenseNet, [6,12,24,16], growth_rate=32,
        num_classes=num_classes) 
    

    
    
def get_model(model_name):
    
    if model_name == "mlp":
        
        return Mlp()
    
    if model_name == "resnet34_10":
    
        return ResNet([3, 4, 6, 3], num_classes=10)
    
    if model_name == "resnet34_100":
    
        return ResNet([3, 4, 6, 3], num_classes=100)
    
    if model_name == "densenet10":
    
        return DenseNet121(num_classes=10)
    
    if model_name == "densenet100":
    
        return DenseNet121(num_classes=100)
    
    
##############################################################################
### Graphing
##############################################################################  

def show_loss_acc_graph(opt_out_list, graph_title):

    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.grid(True)
    #plt.ylim([0, 0.1])
    #plt.xlim([0, 20])
    
    legend_list = []

    for opt_out in opt_out_list:
        legend_list.append('{}: lr={}'.format(opt_out['name'], opt_out['lr']))

    for opt_out in opt_out_list:
        plt.semilogy(opt_out['train_loss'], linewidth=1)#, color='black', marker='s', markevery=5, markersize=5)

    plt.legend(legend_list, loc='upper right')


    plt.title(graph_title)
    plt.xlabel('Epochs')
    plt.ylabel('Training - Softmax Loss (log)')


    plt.subplot(1,2,2)
    plt.grid(True)
    #plt.ylim([0.96, 0.99])
    #plt.xlim([0, 50])


    for opt_out in opt_out_list:
        plt.plot(opt_out['test_acc'], linewidth=1)#, color='black', marker='s', markevery=5, markersize=5)

    plt.legend(legend_list, loc='lower right')


    plt.title(graph_title)
    plt.xlabel('Epochs')
    plt.ylabel('Test - Accuracy')

def show_time_graph(opt_out_list, graph_title):

    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.grid(True)
    #plt.xlim([0, 400])
    #plt.ylim([0, 0.1])
    
    legend_list = []

    for opt_out in opt_out_list:
        legend_list.append('{}: lr={}'.format(opt_out['name'], opt_out['lr']))
    

    for opt_out in opt_out_list:
        opt_time = []
        cumulative = 0

        for i in range(len(opt_out['run_time'])):
            cumulative += opt_out['run_time'][i]
            opt_time.append(cumulative)
        plt.semilogy(opt_time, opt_out['train_loss'], linewidth=1)#, color='black', marker='s', markevery=5, markersize=5)

    plt.legend(legend_list, loc='upper right')


    plt.title(graph_title)
    plt.xlabel('Run Time (s)')
    plt.ylabel('Training - Softmax Loss (log)')


    plt.subplot(1,2,2)
    plt.grid(True)
    #plt.ylim([0.96, 0.99])
    #plt.xlim([0, 50])


    for opt_out in opt_out_list:
        opt_time = []
        cumulative = 0

        for i in range(len(opt_out['run_time'])):
            cumulative += opt_out['run_time'][i]
            opt_time.append(cumulative)
        plt.plot(opt_time, opt_out['test_acc'], linewidth=1)#, color='black', marker='s', markevery=5, markersize=5)


    plt.legend(legend_list, loc='lower right')


    plt.title(graph_title)
    plt.xlabel('Time (s)')
    plt.ylabel('Test - Accuracy')

