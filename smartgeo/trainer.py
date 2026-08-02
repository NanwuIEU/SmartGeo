import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
# from torch.cuda.amp import autocast
import torch.nn as nn
import torch.nn.functional as F
import pickle

import numpy as np

def freeze_module(m):
    for p in m.parameters():
        p.requires_grad = False

def unfreeze_module(m):
    for p in m.parameters():
        p.requires_grad = True

def load_balancing_loss(probs, num_experts):
    """
    probs: (batch_size, num_experts) 
    """

    mean_probs = probs.mean(dim=0).clamp_min(1e-9)  # (num_experts,)
    target = torch.ones_like(mean_probs) / num_experts

    loss = F.kl_div(mean_probs.log(), target, reduction='batchmean')
    # loss = F.mse_loss(mean_probs, target)             
    return loss

def routing_consistemcy_loss(p1, p2):
    p1 = p1.mean(dim=0).clamp_min(1e-9)
    p2 = p2.mean(dim=0).clamp_min(1e-9)
    loss = F.kl_div(p1.log(), p2, reduction='batchmean') + F.kl_div(p2.log(), p1, reduction='batchmean')
    return loss

def train(train_config, model, dataloader, loss_function, optimizer, epoch, scheduler=None, scaler=None):

    # set model train mode
    model.train()
    

    losses = AverageMeter()
    
    # wait before starting progress bar
    time.sleep(0.1)

    # Zero gradients for first step
    optimizer.zero_grad(set_to_none=True)
    
    step = 1
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    for query, reference, ids in bar:
        
        if scaler:
            with torch.amp.autocast(device_type='cuda'):
            
                # data (batches) to device   
                query = query.to(train_config.device)
                reference = reference.to(train_config.device)
            
                # Forward pass
                features1, features2 = model(query, reference)

                if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                    loss = loss_function(features1[0], features2[0], model.module.logit_scale.exp())
                else:
                    loss = loss_function(features1[0], features2[0], model.logit_scale.exp()) 

                loss = loss + (30 - 0.6 * epoch) * ( routing_consistemcy_loss(features1[1], features2[1]) + load_balancing_loss(features1[1], 4) + load_balancing_loss(features2[1], 4)) 
                losses.update(loss.item())
                                 
            scaler.scale(loss).backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad) 
                                
            
            # Update model parameters (weights)
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
   
        else:
        
            # data (batches) to device   
            query = query.to(train_config.device)
            reference = reference.to(train_config.device)

            # Forward pass
            features1, features2 = model(query, reference)
            if torch.cuda.device_count() > 1 and len(train_config.gpu_ids) > 1: 
                loss = loss_function(features1, features2, model.module.logit_scale.exp())
            else:
                loss = loss_function(features1, features2, model.logit_scale.exp()) 
            losses.update(loss.item())

            # Calculate gradient using backward pass
            loss.backward()
            
            # Gradient clipping 
            if train_config.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), train_config.clip_grad)                  
            
            # Update model parameters (weights)
            optimizer.step()
            # Zero gradients for next step
            optimizer.zero_grad()
            
            # Scheduler
            if train_config.scheduler == "polynomial" or train_config.scheduler == "cosine" or train_config.scheduler ==  "constant":
                scheduler.step()
        
        
        
        if train_config.verbose:
            
            monitor = {"loss": "{:.4f}".format(loss.item()),
                       "loss_avg": "{:.4f}".format(losses.avg),
                       "lr" : "{:.6f}".format(optimizer.param_groups[0]['lr'])}
            
            bar.set_postfix(ordered_dict=monitor)
        
        step += 1

    if train_config.verbose:
        bar.close()

    return losses.avg


def predict_query(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    features = {}
    with torch.no_grad():
        
        for img, ids, img_dirs in bar:
        
            ids_list.append(ids)
            
            with torch.amp.autocast(device_type='cuda'):
         
                img = img.to(train_config.device)
                img_feature = model(img1=img, img2=None)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids_list

def predict_reference(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    features = {}
    with torch.no_grad():
        
        for img, ids, img_dirs in bar:
        
            ids_list.append(ids)
            
            with torch.amp.autocast(device_type='cuda'):
         
                img = img.to(train_config.device)
                img_feature = model(img1=None, img2=img)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids_list


def predict(train_config, model, dataloader):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    features = {}
    with torch.no_grad():
        
        for img, ids, img_dirs in bar:
        
            ids_list.append(ids)
            
            with torch.amp.autocast(device_type='cuda'):
         
                img = img.to(train_config.device)
                img_feature, _ = model(img)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids_list



def predict_vigor(train_config, model, dataloader, mode):
    model.eval()

    # Pre-allocate memory for efficiency (assuming fixed batch size and ids_current shape)
    total_samples = len(dataloader.dataset)
    img_features = torch.zeros((total_samples, 4096), dtype=torch.float32, device=train_config.device)
    if mode == 'reference':
        ids = torch.zeros(total_samples, dtype=torch.long, device=train_config.device)
    else:
        ids = torch.zeros((total_samples,4), dtype=torch.long, device=train_config.device)
    # Assuming each id_current has 4 elements, adjust the dimension for ids

    with torch.no_grad():
        for i, (img, ids_current, img_dirs) in enumerate(tqdm(dataloader)):
            img = img.to(train_config.device)
            img_feature, _ = model(img)

            # Normalize is calculated in fp32
            if train_config.normalize_features:
                img_feature = F.normalize(img_feature, dim=-1)

            img_features[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = img_feature
            # Directly assign the 2D ids_current to the corresponding slice in ids
            ids[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = ids_current

    return img_features, ids
