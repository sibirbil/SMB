#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
from torch.optim import Optimizer
from typing import Iterable

from .utils import get_grad_list, compute_grad_norm



##############################################################################
# SMB Optimizer
##############################################################################

class SMB(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1,
        eta: float = 0.99,
        c: float = 0.1,
        independent_batch = False,
        n_batches_per_epoch: int = 500,
        autoschedule = False,
        gamma: float = 0.05, 
        beta: float = 0.9,
        lr_max: float = 20
    ):
        defaults = dict(lr=lr, 
                        eta=eta, 
                        c=c, 
                        independent_batch=independent_batch, 
                        n_batches_per_epoch=n_batches_per_epoch,
                        autoschedule=autoschedule,
                        gamma=gamma,
                        beta=beta,
                        lr_max=lr_max
                        )
        super().__init__(params, defaults)
        
        
    def step(self, closure):
        
        independent_batch = False
        
        for group in self.param_groups:
            
             independent_batch =  group["independent_batch"]
            
        if independent_batch:
            
            return self.step_smbi(closure)
        
        else:
            
            return self.step_smb(closure)
            
        
    def step_smb(self, closure):
        
        if len(self.state) == 0:
            self.state['step'] = 0
            self.state['model_step_list'] = []
            
        loss = closure()
        loss.backward()
        
        for group in self.param_groups:
            
            lr =  group["lr"]  
            eta =  group["eta"]
            c = group["c"]
            
            params = group["params"]
            
            # Auto-scheduling parameters
            gamma = group["gamma"]
            beta = group["beta"]
            lr_max = group["lr_max"]
            
            
            grad_current = get_grad_list(params)
            grad_norm = compute_grad_norm(grad_current)
            
            cond = loss.item() - c * lr * grad_norm.pow(2).item()
            
            
            for p in params:
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state["grad_old"] = torch.zeros_like(p.grad)
                    state["s_old"] = torch.zeros_like(p.data)
                    
                
                if p.grad is None:
                    continue    
                    
                grad = p.grad.data
                state["grad_old"] = grad.clone().detach()
                
                if grad.is_sparse:
                    raise RuntimeError("SMB does not support sparse gradients")
                
                s_new = grad.mul(-lr)
                state["s_old"] = s_new.clone().detach()
                
                p.data.add_(s_new, alpha=1.0)
                
            
            loss_next = closure()
            
            
            if grad_norm >= 1e-8: 

                if loss_next.item() <= cond:
                    
                    self.state["model_step_list"].append(0)
                    
                else:
                        
                    loss_next.backward()
    
                    for p in group["params"]:
    
                        state = self.state[p]
    
                        grad = state["grad_old"]
                        grad_t = p.grad.data
    
                        s_old = state["s_old"]
    
                        g = torch.flatten(grad)
                        gt = torch.flatten(grad_t)
                        s = torch.flatten(s_old)
    
                        sg = torch.dot(s,g)  # (v6)
                        sgt = torch.dot(s,gt)
    
                        y_t = grad_t.sub(grad, alpha=1.0) # y^k_t = g_t - g
    
                        y = torch.flatten(y_t)
                        ys = sgt - sg       # v1 
                        ss = torch.dot(s,s) # v2
                        yy = torch.dot(y,y) # v3
                        yg = torch.dot(y,g) # v4
                        gg = torch.dot(g,g) # v5
    
                        sigma = 0.5*(torch.sqrt(ss)*(torch.sqrt(yy)+torch.sqrt(gg)/eta)-ys)
                        theta = (ys + 2.0*sigma)**2.0 - ss*yy
    
                        cg = -ss/(2.0*sigma)                       # cg(sigma)
                        cs = cg/theta*(-(ys + 2.0*sigma)*yg+yy*sg) # cs(sigma)
                        cy = cg/theta*(-(ys + 2.0*sigma)*sg+ss*yg) # cy(sigma)
    
                        s_new = s_old.mul(cs).add(grad, alpha=cg).add(y_t, alpha=cy)
                            
                        p.data.sub_(s_old, alpha=1.0).add_(s_new, alpha=1.0)    
                      
                    self.state["model_step_list"].append(1)
            
            else:
                
                self.state["model_step_list"].append(0)
                
            #Auto-scheduling
            if group["autoschedule"] and len(self.state['model_step_list']) >= group["n_batches_per_epoch"]:
                if sum(self.state['model_step_list']) / len(self.state['model_step_list']) > gamma:
                    group["lr"] *= beta
                elif group["lr"]/beta <= lr_max:
                    group["lr"] /= beta
                self.state['model_step_list'] = []
                
                     
        self.state['step'] += 1
        
                
        return loss
    
    
    def step_smbi(self, closure):
        
        if len(self.state) == 0:
            self.state['step'] = 0
            self.state['model_step'] = False
            self.state['model_step_list'] = []
            
        loss = closure()
        loss.backward()
            
        
        for group in self.param_groups:
            
            lr =  group["lr"]  
            eta =  group["eta"]
            c = group["c"]
            
            params = group["params"]
            
            # Auto-scheduling parameters
            gamma = group["gamma"]
            beta = group["beta"]
            lr_max = group["lr_max"]
            
            
            grad_current = get_grad_list(params)
            grad_norm = compute_grad_norm(grad_current)
            
            cond = loss.item() - c * lr * grad_norm.pow(2).item()
            
            
            if not self.state['model_step']:
                
                self.state["model_step_list"].append(0)

                for p in params:

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["grad_prev"] = torch.zeros_like(p.grad)
                        state["grad_old"] = torch.zeros_like(p.grad)
                        state["s_old"] = torch.zeros_like(p.data)

                    if p.grad is None:
                        continue    

                    grad = p.grad.data
                    state["grad_prev"] = grad.clone().detach()

                    if grad.is_sparse:
                        raise RuntimeError("SMB does not support sparse gradients")

                    s_new = grad.mul(-lr)
                    state["s_old"] = s_new.clone().detach()

                    p.data.add_(s_new, alpha=1.0)

                loss_next = closure()
        

                if grad_norm >= 1e-8: 

                    if loss_next.item() > cond:
                        
                        for p in params:
                            state = self.state[p]
                            p.data.sub_(state["s_old"], alpha=1.0)

                        self.state['model_step'] = True
                    
             
            else:
                
                self.state["model_step_list"].append(1)
                
                for p in params:

                    state = self.state[p]

                    if p.grad is None:
                        continue    

                    grad = p.grad.data
                    state["grad_old"] = grad.clone().detach()

                    if grad.is_sparse:
                        raise RuntimeError("SMB does not support sparse gradients")

                    s_new = grad.mul(-lr)
                    state["s_old"] = s_new.clone().detach()

                    p.data.add_(s_new, alpha=1.0)

                loss_next = closure()
                loss_next.backward()

                for p in group["params"]:

                    state = self.state[p]
                    
                    grad_prev = state["grad_prev"]
                    grad = state["grad_old"]
                    grad_t = p.grad.data
                    s_old = state["s_old"]
                    
                    y_t = grad_t.sub(grad, alpha=1.0) # y^k_t = g_t - g
                    
                    g_prev = torch.flatten(grad_prev)
                    g = torch.flatten(grad)
                    gt = torch.flatten(grad_t)
                    s = torch.flatten(s_old)
                    y = torch.flatten(y_t)

                    
                    ys = torch.dot(y,s)      # v1
                    ss = torch.dot(s,s)      # v2
                    yy = torch.dot(y,y)      # v3
                    yg = torch.dot(y,g_prev) # v4
                    gg = torch.dot(g,g)      # v5
                    sg = torch.dot(s,g_prev) # (v6)

                    sigma = 0.5*(torch.sqrt(ss)*(torch.sqrt(yy)+torch.sqrt(gg)/eta)-ys)
                    theta = (ys + 2.0*sigma)**2.0 - ss*yy

                    cg = -ss/(2.0*sigma)                       # cg(sigma)
                    cs = cg/theta*(-(ys + 2.0*sigma)*yg+yy*sg) # cs(sigma)
                    cy = cg/theta*(-(ys + 2.0*sigma)*sg+ss*yg) # cy(sigma)

                    s_new = s_old.mul(cs).add(grad_prev, alpha=cg).add(y_t, alpha=cy)
                        
                    p.data.sub_(state["s_old"], alpha=1.0).add_(s_new, alpha=1.0)
                 
                self.state['model_step'] = False
                
            #Auto-scheduling
            if group["autoschedule"] and len(self.state['model_step_list']) >= group["n_batches_per_epoch"]:
                if sum(self.state['model_step_list']) / len(self.state['model_step_list']) > gamma/2:
                    group["lr"] *= beta
                elif group["lr"]/beta <= lr_max:
                    group["lr"] /= beta
                self.state['model_step_list'] = []
                
        self.state['step'] += 1
                
        return loss
    
    
    



