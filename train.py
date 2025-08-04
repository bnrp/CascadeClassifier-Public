import random
import os
import sys
import pickle

import timm

import torch
import torchvision
from torchvision.transforms import v2

import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tqdm

import utils.self_utils as self_utils

import MSA_CC
import load
import loadCUB as loadcub


def train(model, optimizer, criterion, steps, train, test, test_steps, epochs, device, normalized_labels=None, scheduler=None, g_acc=16, save=False):
    losses = torch.zeros(epochs).to(device)
    model.train()
    cm = None

    for epoch in range(epochs):
        ls = 0
        for i, (X,y) in zip(tqdm.trange(steps), train):
            X = torch.Tensor(X).to(device)
            label = normalized_labels[y]

            if batch_size == 1: label = np.array([label])

            y1 = torch.Tensor(label[:,0]).to(torch.long).to(device) 
            y2 = torch.Tensor(label[:,1]).to(torch.long).to(device) 
            y3 = torch.Tensor(label[:,2]).to(torch.long).to(device) 
            y4 = torch.Tensor(label[:,3]).to(torch.long).to(device) 

            y1 = torch.nn.functional.one_hot(y1, num_classes=2).to(torch.float32)
            y2 = torch.nn.functional.one_hot(y2, num_classes=50).to(torch.float32)
            y3 = torch.nn.functional.one_hot(y3, num_classes=404).to(torch.float32)
            y4 = torch.nn.functional.one_hot(y4, num_classes=555).to(torch.float32)

            pred = model.forward(X)

            loss1 = criterion[0](pred[0], y1)
            loss2 = criterion[1](pred[1], y2)
            loss3 = criterion[2](pred[2], y3)
            loss4 = criterion[3](pred[3], y4)

            loss = loss1 + loss2 + loss3 + loss4
            loss.backward()

            if (i % g_acc) == 0:
                for opt in optimizer:
                    opt.step()
                    opt.zero_grad(set_to_none=True)

            ls += loss.item()
            if scheduler is not None: scheduler.step()
        
        losses[epoch] = ls / steps
        print('Epoch ' + str(epoch + 1) + '/' + str(epochs) + ' Loss: ' + str(losses[epoch].item()))

        if epoch > 5:
            cm = evaluate(model, test, test_steps, device, normalized_labels)
            model.train()
            print('-'*50)
            print('Passerine v. Nonpasserine')
            print(cm[0])
            print('C1 Accuracy = ' + str(torch.sum(torch.diag(cm[0])).item() / torch.sum(cm[0]).item()))
            
            print('-'*50)
            print('Type')
            print(cm[1])
            print('C2 Accuracy = ' + str(torch.sum(torch.diag(cm[1])).item() / torch.sum(cm[1]).item()))
            
            print('-'*50)
            print('Species')
            print(cm[2])
            print('C3 Accuracy = ' + str(torch.sum(torch.diag(cm[2])).item() / torch.sum(cm[2]).item()))
            
            print('-'*50)
            print('sub-Species')
            print(cm[3])
            print('C4 Accuracy = ' + str(torch.sum(torch.diag(cm[3])).item() / torch.sum(cm[3]).item()))
            print('-'*50)

    if save:
        torch.save(model.state_dict(), 'Saved_Models/MSA_CC_Acc_' + str(torch.sum(torch.diag(cm[3])).item() / torch.sum(cm[3]).item()) + '.pt')

    return losses


def train_cub(model, optimizer, criterion, steps, train, test, test_steps, epochs, device, scheduler=None, g_acc=16, save=False):
    losses = torch.zeros(epochs).to(device)
    model.train()
    cm = None

    for epoch in range(epochs):
        ls = 0
        for i, (X,y) in zip(tqdm.trange(steps), train):
            X = torch.Tensor(X).to(device)
            y = torch.Tensor(y).to(device)

            y = torch.nn.functional.one_hot(y, num_classes=200).to(torch.float32)

            pred = model.forward(X)

            loss = criterion[0](pred, y)
                    
            loss.backward()

            if (i % g_acc) == 0:
                for opt in optimizer:
                    opt.step()
                    opt.zero_grad(set_to_none=True)

            ls += loss.item()
            if scheduler is not None: scheduler.step()
        
        losses[epoch] = ls / steps
        print('Epoch ' + str(epoch + 1) + '/' + str(epochs) + ' Loss: ' + str(losses[epoch].item()))

        if epoch >= 0:
            cm = evaluate_cub(model, test, test_steps, device)
            model.train()
            
            print('Species')
            print(cm)
            print('C3 Accuracy = ' + str(torch.sum(torch.diag(cm)).item() / torch.sum(cm).item()))

    if save:
        torch.save(model.state_dict(), 'Saved_Models/MSA_CC_Acc_' + str(torch.sum(torch.diag(cm)).item() / torch.sum(cm).item()) + '.pt')

    return losses


def evaluate_cub(model, test, steps, device='cuda:0'):
    model.eval()

    cm = torch.zeros(200, 200).to(device)
    
    good = total = 0
    with torch.no_grad():
        for _, (X, y) in zip(tqdm.trange(steps), test):
            X = torch.Tensor(X).to(device)
            y = torch.Tensor(y).to(device)
        
            pred = model(X)

            if type(pred) == list:
                pred1 = torch.argmax(pred, dim=-1)

                idx = torch.cat((pred1.to(torch.long).unsqueeze(0), y.to(torch.long).unsqueeze(0)), 0)
                unique, counts = torch.unique(idx, dim=1, return_counts=True)
    
                cm[unique[0], unique[1]] += counts1
            else:
                pred4 = torch.argmax(pred, dim=-1)

                idx4 = torch.cat((pred4.to(torch.long).unsqueeze(0), y.to(torch.long).unsqueeze(0)), 0)
                unique4, counts4 = torch.unique(idx4, dim=1, return_counts=True)
                
                cm[unique4[0], unique4[1]] += counts4
            
    return cm


def evaluate(model, test, steps, device='cuda:0', normalized_labels=None):
    model.eval()

    cm = [
            torch.zeros(2, 2).to(device),
            torch.zeros(50, 50).to(device),
            torch.zeros(404, 404).to(device),
            torch.zeros(555, 555).to(device)
    ]
    
    good = total = 0
    with torch.no_grad():
        for _, (X, y) in zip(tqdm.trange(steps), test):
            X = torch.Tensor(X).to(device)
            label = normalized_labels[y]
            if batch_size == 1: label = np.array([label])
            y1 = torch.Tensor(label[:,0]).to(device) 
            y2 = torch.Tensor(label[:,1]).to(device) 
            y3 = torch.Tensor(label[:,2]).to(device) 
            y4 = torch.Tensor(label[:,3]).to(device) 
        
            pred = model(X)

            if type(pred) == list:
                pred1 = torch.argmax(pred[0], dim=-1)
                pred2 = torch.argmax(pred[1], dim=-1)
                pred3 = torch.argmax(pred[2], dim=-1)
                pred4 = torch.argmax(pred[3], dim=-1)

                idx1 = torch.cat((pred1.to(torch.long).unsqueeze(0), y1.to(torch.long).unsqueeze(0)), 0)
                unique1, counts1 = torch.unique(idx1, dim=1, return_counts=True)
                idx2 = torch.cat((pred2.to(torch.long).unsqueeze(0), y2.to(torch.long).unsqueeze(0)), 0)
                unique2, counts2 = torch.unique(idx2, dim=1, return_counts=True)
                idx3 = torch.cat((pred3.to(torch.long).unsqueeze(0), y3.to(torch.long).unsqueeze(0)), 0)
                unique3, counts3 = torch.unique(idx3, dim=1, return_counts=True)
                idx4 = torch.cat((pred4.to(torch.long).unsqueeze(0), y4.to(torch.long).unsqueeze(0)), 0)
                unique4, counts4 = torch.unique(idx4, dim=1, return_counts=True)
    
                cm[0][unique1[0], unique1[1]] += counts1
                cm[1][unique2[0], unique2[1]] += counts2
                cm[2][unique3[0], unique3[1]] += counts3
                cm[3][unique4[0], unique4[1]] += counts4
            else:
                pred4 = torch.argmax(pred, dim=-1)

                idx4 = torch.cat((pred4.to(torch.long).unsqueeze(0), y4.to(torch.long).unsqueeze(0)), 0)
                unique4, counts4 = torch.unique(idx4, dim=1, return_counts=True)
                
                cm[3][unique4[0], unique4[1]] += counts4
            
    return cm


def run_training(epochs=10, batch_size=2, mtype='21k_b_big', data='Given'):
    load.set_seeds()
    device = load.setup_cuda()

    g_acc = 1
    if batch_size < 32:
        g_acc = int(32 / batch_size)

    hierarchical_labels, normalized_labels, normalized_class_names = load.load_label_helpers()

    if data == 'Given':
        train_dataloader, test_dataloader, train_size, test_size = load.load_original_data(batch_size=batch_size)
        c1_freq, c2_freq, c3_freq, c4_freq = load.load_orig_class_freqs()
    else:
        train_dataloader, test_dataloader, train_size, test_size = load.load_data(batch_size=batch_size)
        c1_freq, c2_freq, c3_freq, c4_freq = load.load_class_freqs()

    #adam_lr = 0.000025
    adam_lr = 0.000025
    steps = int(train_size // batch_size)
    test_steps = int(test_size // batch_size)
    max_T = np.floor(steps*25*0.5)

    model = load.load_model(mtype=mtype).to(device)

    optimizer = [torch.optim.AdamW(model.parameters(), lr=adam_lr, betas=(0.90,0.95))]

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[0], T_max=max_T, eta_min=0.00000000000001, last_epoch=-1)

    criterion = [torch.nn.CrossEntropyLoss(weight=c1_freq.to(device)),
                 torch.nn.CrossEntropyLoss(weight=c2_freq.to(device)),
                 torch.nn.CrossEntropyLoss(weight=c3_freq.to(device)),
                 torch.nn.CrossEntropyLoss(weight=c4_freq.to(device))]
    c1_freq = c2_freq = c3_freq = c4_freq = None
    
    #i = 0
    #for name, param in model.named_parameters():
    #    print(i, name, param.requires_grad)
    #    i += 1

    losses = train(
                    model, 
                    optimizer, 
                    criterion, 
                    steps, 
                    train_dataloader, 
                    test_dataloader, 
                    test_steps, 
                    epochs,
                    device=device, 
                    normalized_labels=normalized_labels, 
                    scheduler=scheduler,
                    g_acc=g_acc,
                    save=True)


def run_training_cub(epochs=10, batch_size=2, mtype='21k_b_big', data='Given'):
    load.set_seeds()
    device = load.setup_cuda()

    g_acc = 1
    if batch_size < 32:
        g_acc = int(32 / batch_size)

    train_dataloader, test_dataloader, train_size, test_size = loadcub.load_original_data(batch_size=batch_size)

    #adam_lr = 0.000025
    adam_lr = 0.000075
    steps = int(train_size // batch_size)
    test_steps = int(test_size // batch_size)
    max_T = np.floor(steps*25*0.5)

    model = loadcub.load_model(mtype=mtype).to(device)

    optimizer = [torch.optim.AdamW(model.parameters(), lr=adam_lr, betas=(0.90,0.95))]

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[0], T_max=max_T, eta_min=0.00000000000001, last_epoch=-1)

    criterion = [torch.nn.CrossEntropyLoss()]
    
    #i = 0
    #for name, param in model.named_parameters():
    #    print(i, name, param.requires_grad)
    #    i += 1

    losses = train_cub(
                    model, 
                    optimizer, 
                    criterion, 
                    steps, 
                    train_dataloader, 
                    test_dataloader, 
                    test_steps, 
                    epochs,
                    device=device, 
                    scheduler=scheduler,
                    g_acc=g_acc,
                    save=True)


if __name__ == '__main__':
    with torch.no_grad():
        torch.cuda.empty_cache()
    epochs = 10
    batch_size = 1
    run_training_cub(epochs=epochs, batch_size=batch_size, mtype='21k_b_big')
