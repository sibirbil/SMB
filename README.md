### SMB

Sochastic gradient descent with model building. Train your network with a fast and robust optimizer. 

  
 
#### 1. Installation. 

`pip install git+https://github.com/sbirbil/SMB.git`


#### 2. Usage. 
Essentially, the algorithm progress by executing a step via `SGD` then checks the Armijo conditions: in case they are not satisfied it builds an auxillary model to tune the learning rate and the direction of the next step accordingly. This model based update can be done in two ways: either by using the current batch or by taking another *independent* batch. The user has to choose this option by initializing the hyperparameter `independent_batch`. Note that SMB needs also a closure function.

Use SMB in your codeas follows (see `smb/tutorial.ipynb` for a complete example). Set the hyperparameter `independent_batch` to `True` in order to use the `SMBi` optimizer.

```python

import smb

 
optimizer = smb.SMB(model.parameters(), independent_batch=False) #independent_batch=True for SMBi optimizer


for epoch in range(100):
    
    # training steps
    model.train()
    
    for batch_index, (data, target) in enumerate(train_loader):
            
        # create loss closure for smb algorithm
        def closure():
            optimizer.zero_grad()
            loss = torch.nn.CrossEntropyLoss((model(data), target)
            return loss
        
        # forward pass
        loss = optimizer.step(closure=closure)
```
