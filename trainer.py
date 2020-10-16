'''trainer'''

import os
import sys
import signal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import warnings

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import argparse

from model import SeismoNet
from dataloader import CEBSDataset

warnings.filterwarnings("ignore")


def dump_and_exit(signalnumber, frame):
    if not(os.path.exists("best_model")):
           os.mkdir("best_model")
    torch.save(model_state, "best_model/best_model_on_SIGINT.pt")
    sys.exit(0)
    
def main(args):
    

    global model_state
    
    test_size = float(args.test_size)
    val_size = float(args.val_size)
    data_path = os.path.join(args.data_path, args.file_type)
    lr = float(args.lr)
    train_batch_size = int(args.train_batch_size)
    val_batch_size = int(args.val_batch_size)
    epochs = args.epochs
    typ = args.file_type
    
    print ("Training SeismoNet on CEBS")
    print (args)

    data = torch.load(os.path.join(data_path,'data.pt'))
    target = torch.load(os.path.join(data_path,'labels.pt'))
    print ("total no.of windows", len(data))
    x_train, x_val, y_train, y_val = train_test_split(data, target, random_state = 42, test_size = val_size + test_size)
    x_val,x_test, y_val,y_test = train_test_split(x_val,y_val, random_state = 32, test_size = (test_size/(test_size + val_size)))
    train, val, test = TensorDataset(x_train, y_train), TensorDataset(x_val, y_val), TensorDataset(x_test, y_test)
    
    train_loader = DataLoader(train, batch_size=train_batch_size, shuffle =False)
    val_loader = DataLoader(val, batch_size = val_batch_size, shuffle = False)
    test_loader = DataLoader(test, batch_size = 1 , shuffle = False)
    

    
    writer = SummaryWriter()
    
    model= SeismoNet(x_train.shape).cuda()
    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[50,150,250], gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    best_loss = 1000
    best_accuracy = 0
    if not(os.path.exists("best_model/")):
           os.mkdir("best_model/")
    
    for epoch in range(int(epochs)):

        model.train()
        print('epochs {}/{} '.format(epoch+1,epochs))
        running_loss = 0.0
        running_loss_v = 0.0
        correct = 0
        correct_v = 0
        for idx, (inputs,labels) in tqdm(enumerate(train_loader), total = len(train_loader)):

            inputs = inputs.cuda()
            labels = labels.cuda()
            
            
            optimizer.zero_grad()
            
            y_pred= model(inputs)


            loss = criterion(y_pred,labels) 
            running_loss += loss
            loss.backward()
            optimizer.step()
            

        scheduler.step()
        model.eval()
        with torch.no_grad():
            for idx,(inputs_v,labels_v) in tqdm(enumerate(val_loader),total=len(val_loader)):
                
                inputs_v = inputs_v.cuda()
                labels_v = labels_v.cuda()
                y_pred_v = model(inputs_v)
                loss_v = criterion(y_pred_v,labels_v)

                running_loss_v += loss_v
                

            val_loss = running_loss_v/len(val_loader)
            model_state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_loss': val_loss
            }
           
            if (val_loss <= best_loss):
                best_loss = running_loss_v/len(val_loader)
                out = torch.save(model_state, f='best_model/best_model.pt')        
            
        print('train loss: {:.4f} val loss : {:.4f}'.format(running_loss/len(train_loader), running_loss_v/len(val_loader)))  
        writer.add_scalar("Loss/train_loss",running_loss/len(train_loader), epoch )
        writer.add_scalar("Loss/val_loss",running_loss_v/len(val_loader), epoch )
        
    
    writer.close()
         
    print ("Completed")
    torch.save(model_state, f='best_model/best_model_training_completed.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',nargs="?", const = "saved_data/", default = "saved_data/", help = 'Path to saved files directory')
    parser.add_argument('--file_type',nargs="?", const = "b", default = "b", help = "file type")
    parser.add_argument('--test_size',  nargs='?',const = 0.2,default = 0.2,  help = 'Size of Test Set (float)')
    parser.add_argument('--val_size',nargs='?', const = 0.2,default = 0.2, help = 'Size of Validation Set (float)')
    parser.add_argument('--train_batch_size',nargs='?', const = 64,default = 64, help = 'Batch Size of Train Loader')
    parser.add_argument('--val_batch_size',nargs='?', const = 64,default = 64, help = 'Batch Size of Validation Loader')
    parser.add_argument('--epochs',nargs='?', const = 150,default = 150, help = 'Number of Epochs')
    parser.add_argument("--lr",nargs = "?",const = 0.001,default = 0.001, help = 'Learning Rate')
    signal.signal(signal.SIGINT, dump_and_exit)
    args = parser.parse_args()
    main(args)



