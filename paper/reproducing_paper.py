#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.optim as optim

import time
import datetime
import numpy as np
import json

import smb
from smb import utils as ut
import sls  # Install sls from https://github.com/IssamLaradji/sls


##############################################################################
# OPTIONS
##############################################################################


# Epochs to train for
epochs = 200

# Dataset-Model
TrainOptions = {1:('mnist', 'mlp'), 
                2:('cifar10', 'resnet34_10'), 
                3:('cifar10', 'densenet10'), 
                4:('cifar100', 'resnet34_100'), 
                5:('cifar100', 'densenet10')
                }
dataset_name, model_name = TrainOptions[1]


# Batch Size
batch_size = 128


##############################################################################

# Check if GPU is available
use_GPU = torch.cuda.is_available()


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
    

# Get Dataset
train_set, test_set, train_loader = ut.get_dataset(dataset_name, batch_size)
n_batches_per_epoch = len(train_loader)



##############################################################################
# Train with SMB optimizer
##############################################################################

independent_batch = False
autoschedule = False

opt_out = {}

if independent_batch:
    opt_out.update({'name':'SMBi'})
else:
    opt_out.update({'name':'SMB'})

opt_out.update({'independent_batch':independent_batch, 
           'autoschedule':autoschedule, 
           'gamma':0.05, 
           'beta':0.9, 
           'lr':1, 
           'c':0.1, 
           'eta':0.99, 
           'data':dataset_name, 
           'model':model_name, 
           })

# Get Model
model = ut.get_model(model_name)
if use_GPU:
    model.cuda()
 
 
# loss function
criterion = ut.softmax_loss
 
optimizer = smb.SMB(model.parameters(), 
                lr=opt_out['lr'], 
                c=opt_out['c'], 
                eta=opt_out['eta'], 
                independent_batch=opt_out['independent_batch'], 
                autoschedule=opt_out['autoschedule'],
                n_batches_per_epoch=n_batches_per_epoch)

print('\n' + 'Starting to train with {} optimizer: For {} epochs'.format(opt_out['name'], epochs))

train_loss_list = []
test_acc_list = []
run_time_list = []


for epoch in range(1, epochs+1):
    
    step_type = []
        
    begin = time.time()
    
    # training steps
    model.train()
    
    for batch_index, (data, target) in enumerate(train_loader):
        
        # moves tensors to GPU if available
        if use_GPU:
            data, target = data.cuda(), target.cuda() 
            
        # create loss closure for smb algorithm
        def closure():
            optimizer.zero_grad()
            loss = criterion(model, data, target)
            return loss
        
        # forward pass
        loss = optimizer.step(closure=closure)
        
    end = time.time()
        
    train_loss = ut.compute_loss(model, train_set)
    test_acc = ut.compute_accuracy(model, test_set)
        
    train_loss_list.append(train_loss)
    test_acc_list.append(test_acc)
    run_time_list.append(end-begin)
        
    # Display loss statistics
    print(f'Epoch: {epoch}   -   Training Loss: {round(train_loss, 6)}  -  Test Accuracy: {round(test_acc, 6)}  -  Time: {round(end-begin, 2)}')

    
opt_out.update({'train_loss':train_loss_list,
                 'test_acc':test_acc_list,
                 'run_time':run_time_list,
                })

now = datetime.datetime.now() # current date and time
date_time = now.strftime("%Y_%m_%d_%H_%M_%S")


filename = "{}_{}_{}_{}.json".format(opt_out['name'],
                                             opt_out['data'], 
                                             opt_out['model'],
                                             date_time
                                             )

with open(filename, 'w') as f:
    json.dump(opt_out, f)

 

##############################################################################
# Train with SLS optimizer
##############################################################################


opt_out = {'name':'SLS', 
           'lr':1, 
           'c':0.1, 
           'reset_option':1, 
           'data':dataset_name, 
           'model':model_name, 
           }

# Get Model
model = ut.get_model(model_name)
if use_GPU:
    model.cuda()

 
# loss function
criterion = ut.softmax_loss


optimizer = sls.Sls(model.parameters(), 
                    init_step_size=opt_out['lr'], 
                    reset_option=opt_out['reset_option'], 
                    c=opt_out['c'], 
                    n_batches_per_epoch=n_batches_per_epoch
                   )

print('\n' + 'Starting to train with {} optimizer: For {} epochs'.format(opt_out['name'], epochs))


train_loss_list = []
train_iter_loss_list = []
test_acc_list = []
run_time_list = []

loss = None

for epoch in range(1, epochs+1):
        
    begin = time.time()
    
    # training steps
    model.train()
    
    for batch_index, (data, target) in enumerate(train_loader):
        
        # moves tensors to GPU
        if use_GPU:
            data, target = data.cuda(), target.cuda() 
            
        # create loss closure for sls algorithm
        closure = lambda :  criterion(model, data, target)  
        # clears gradients
        optimizer.zero_grad()
        
        loss = optimizer.step(closure=closure)
        
        train_iter_loss_list.append(loss.item())
        
    end = time.time()
    
    train_loss = ut.compute_loss(model, train_set)
    test_acc = ut.compute_accuracy(model, test_set)
        
    train_loss_list.append(train_loss)
    test_acc_list.append(test_acc)
    run_time_list.append(end-begin)
        
    # Display loss statistics
    print(f'Epoch: {epoch}   -   Training Loss: {round(train_loss, 6)}  -  Test Accuracy: {round(test_acc, 6)}  -  Time: {round(end-begin, 2)}')
    
    #print(epoch, end=' ')
    
opt_out.update({'train_loss':train_loss_list,
                 'test_acc':test_acc_list,
                 'run_time':run_time_list,
                 'train_iter_loss':train_iter_loss_list,
                })


now = datetime.datetime.now() # current date and time
date_time = now.strftime("%Y_%m_%d_%H_%M_%S")


filename = "{}_{}_{}_{}.json".format(opt_out['name'],
                                             opt_out['data'], 
                                             opt_out['model'],
                                             date_time
                                             )

with open(filename, 'w') as f:
    json.dump(opt_out, f)
    







##############################################################################
# Train with Adam optimizer
##############################################################################

opt_out = {'name':'Adam', 
           'lr':0.001, 
           'data':dataset_name, 
           'model':model_name,
           } 
 
# Get Model
model = ut.get_model(model_name)
if use_GPU:
    model.cuda()

# loss function
criterion = ut.softmax_loss

# optimizer
optimizer = optim.Adam(model.parameters(), lr = opt_out['lr'])


print('\n' + 'Starting to train with {} optimizer: For {} epochs'.format(opt_out['name'], epochs))


train_loss_list = []
test_acc_list = []
run_time_list = []
    

for epoch in range(1, epochs+1):
        
    begin = time.time()

    # training steps
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):            
            
        # moves tensors to GPU
        if use_GPU:
            data, target = data.cuda(), target.cuda()     
        # clears gradients
        optimizer.zero_grad()
        # loss in batch
        loss = criterion(model, data, target)
        # backward pass for loss gradient
        loss.backward()
            
            
        # update paremeters
        optimizer.step()
            
    end = time.time()
    
    
    # Calculate metrics
    train_loss = ut.compute_loss(model, train_set)
    test_acc = ut.compute_accuracy(model, test_set)
    
    train_loss_list.append(train_loss)
    test_acc_list.append(test_acc)
    run_time_list.append(end-begin)
        
    # Display loss statistics
    print(f'Epoch: {epoch}   -   Training Loss: {round(train_loss, 6)}   -   Test Accuracy: {round(test_acc, 6)}  -  Time: {round(end-begin, 2)}')


opt_out.update({'train_loss':train_loss_list,
                 'test_acc':test_acc_list,
                 'run_time':run_time_list,
                })

now = datetime.datetime.now() # current date and time
date_time = now.strftime("%Y_%m_%d_%H_%M_%S")


filename = "{}_{}_{}_{}.json".format(opt_out['name'],
                                             opt_out['data'], 
                                             opt_out['model'],
                                             date_time
                                             )

with open(filename, 'w') as f:
    json.dump(opt_out, f)



##############################################################################
# Train with SGD optimizer
##############################################################################

opt_out = {'name':'SGD', 
           'lr':0.1, 
           'data':dataset_name, 
           'model':model_name,
           }

# Get Model
model = ut.get_model(model_name)
if use_GPU:
    model.cuda()
 
# loss function
criterion = ut.softmax_loss

# optimizer
optimizer = optim.SGD(model.parameters(), lr = opt_out['lr'])

print('\n' + 'Starting to train with {} optimizer: For {} epochs'.format(opt_out['name'], epochs))


train_loss_list = []
test_acc_list = []
run_time_list = []
    

for epoch in range(1, epochs+1):
        
    begin = time.time()

    # training steps
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):            
            
        # moves tensors to GPU
        if use_GPU:
            data, target = data.cuda(), target.cuda()     
        # clears gradients
        optimizer.zero_grad()
        # loss in batch
        loss = criterion(model, data, target)
        # backward pass for loss gradient
        loss.backward()
            
            
        # update paremeters
        optimizer.step()
            
    end = time.time()
    
    
    # Calculate metrics
    train_loss = ut.compute_loss(model, train_set)
    test_acc = ut.compute_accuracy(model, test_set)
    
    train_loss_list.append(train_loss)
    test_acc_list.append(test_acc)
    run_time_list.append(end-begin)
        
    # Display loss statistics
    print(f'Epoch: {epoch}   -   Training Loss: {round(train_loss, 6)}   -   Test Accuracy: {round(test_acc, 6)}  -  Time: {round(end-begin, 2)}')


opt_out.update({'train_loss':train_loss_list,
                 'test_acc':test_acc_list,
                 'run_time':run_time_list,
                })

now = datetime.datetime.now() # current date and time
date_time = now.strftime("%Y_%m_%d_%H_%M_%S")


filename = "{}_{}_{}_{}.json".format(opt_out['name'],
                                             opt_out['data'], 
                                             opt_out['model'],
                                             date_time
                                             )

with open(filename, 'w') as f:
    json.dump(opt_out, f)
    
    
    

 


    