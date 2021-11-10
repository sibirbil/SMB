### SMB

Sochastic gradient descent with model building. Train your network with a fast and robust optimizer. 

  
 
#### 1. Installation. `pip install git+https://github.com/sbirbil/SMB.git`

Then you can import as 
```python
import smb
```

#### 2. Usage. 
Essentially, the algoritm progress by executing a step via `SGD` then checks the Armijo conditions: in case they are not satisfied it builds an auxillary model to tune the learning rate and the direction of the next step accordingly. This model based update can be done in two ways: either on can use the current batch or by taking another *independent* batch. The user has to choose this option by initializing the hyperparameter `independent_batch`. Note that SMB needs also a closure function.

A minimal example of a piece of code to train your model is as follows (see tutorial.ipynb for a complete example)
```python
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
           'lr':0.5, 
           'c':0.1, 
           'eta':0.99, 
           'data':dataset_name, 
           'model':model_name, 
           })

# You have to define your model or you can use one of the models from smb/utils.py by importing 
# from smb import utils as ut
# then set 
# model = ut.get_model(model_name)
if use_GPU:
    model.cuda()
 
 
# loss function
# for instance 
# criterion = ut.softmax_loss
 
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

opt_out_list.append(opt_out)
```
