### SMB

Stochastic gradient descent with model building. Train your network with a fast and robust optimizer.  The details of the algorithm is given in the paper by [Birbil et al. (2021)](http://www.optimization-online.org/DB_HTML/2021/11/8683.html).

  
 
#### 1. Installation. 

`pip install git+https://github.com/sbirbil/SMB.git`


#### 2. Usage. 

Use SMB in your code as follows (see `smb/tutorial.ipynb` for a complete example). Set the hyper-parameter `independent_batch` to `True` in order to use the `SMBi` optimizer. See the paper for more information.

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

#### 3. Experiments. 

See `smb/paper/reproducing_paper.py` in order to reproduce the results in the paper. 
