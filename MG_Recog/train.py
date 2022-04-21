import time
import torch
import os
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader
from numpy import argmax

from MG_Recog.network import *
from MG_Recog.dataloader import *
from MG_Recog.evaluation import *

def train(X, Y, groupsLabel, train_subject, test_subject, epochs=500, lr_features=0.00001, lr_classifier=0.0001, batch_size=256, pretext_epoch=10, train_model=False):
    
    # Create model directory
    os.makedirs("MG_Recog_Weights", exist_ok=True)

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
             
    loss_fn = nn.CrossEntropyLoss()     
            
    # For train and test index
    recog_train_index = []
    recog_test_index = []

    for index, subject_id in enumerate(groupsLabel):
        if int(subject_id) in train_subject:
            recog_train_index.append(index)
        elif int(subject_id) in test_subject:
            recog_test_index.append(index)

    X_train, X_test = [X[i] for i in recog_train_index], [X[i] for i in recog_test_index] 
    y_train, y_test = [Y[i] for i in recog_train_index], [Y[i] for i in recog_test_index] 

    start_time_sec = time.time()
        
    # Write final result to final_result.txt
    final_result = open("MG_Recog_Weights/final_result.txt", "w")
        
    model = LightWeight_Network_Recog(pretext_epoch = pretext_epoch).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.features.parameters(), 'lr': lr_features, 'momentum': 0.9},
        {'params': model.classifier.parameters(), 'lr': lr_classifier, 'momentum': 0.9}
    ])
    
    # Initialize training dataloader
    X_train = torch.Tensor(X_train).permute(0,3,1,2)
    y_train = argmax(torch.Tensor(y_train), axis=-1)
    train_dl = DataLoader(
        OFFRecognitionDataset((X_train, y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    
    # Initialize testing dataloader
    X_test = torch.Tensor(X_test).permute(0,3,1,2)
    y_test = argmax(torch.Tensor(y_test), axis=-1)
    test_dl = DataLoader(
        OFFRecognitionDataset((X_test, y_test)),
        batch_size=batch_size,
        shuffle=False,
    )
    
    # print('Train:', len(X_train), 'Test:', len(X_test))

    if train_model:

        for epoch in range(1, epochs+1):

            # Training
            model.train()
            train_loss         = 0.0
            num_train_correct  = 0
            num_train_examples = 0
            num_top5_correct = 0
                
            for batch in train_dl:
                x    = batch[0].to(device)
                y    = batch[1].to(device)
                optimizer.zero_grad()
                yhat = model(x)
                yhat = yhat.view(len(yhat), 32)

                loss = loss_fn(yhat, y)
                loss.backward()
                optimizer.step()

                train_loss         += loss.data.item() * x.size(0)
                num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
                num_train_examples += x.shape[0]
                if epoch == 1 or epoch % 20 == 0:
                    num_top5_correct   += topKCorrect(y, yhat, 5)

            train_acc      = num_train_correct / num_train_examples
            train_top5_acc = num_top5_correct / num_train_examples
            train_loss     = train_loss / len(train_dl.dataset)

            # Testing
            model.eval()
            val_loss       = 0.0
            num_val_correct  = 0
            num_val_examples = 0
            num_top5_correct = 0
                
            for batch in test_dl:
                x    = batch[0].to(device)
                y    = batch[1].to(device)
                yhat = model(x)
                yhat = yhat.view(len(yhat), 32)
                loss = loss_fn(yhat, y)

                val_loss         += loss.data.item() * x.size(0)
                num_val_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
                num_val_examples += y.shape[0]
                if epoch == 1 or epoch % 20 == 0:
                    num_top5_correct += topKCorrect(y, yhat, 5)

            val_acc      = num_val_correct / num_val_examples
            val_top5_acc = num_top5_correct / num_val_examples
            val_loss     = val_loss / len(test_dl.dataset)
            
            if epoch == 1 or epoch % 20 == 0: # Note that val = test herein
                print('Epoch %3d/%3d, train loss: %5.4f, train top1: %5.4f, train top5: %5.4f, val loss: %5.4f, val top1: %5.4f, val top5: %5.4f' % (epoch, epochs, train_loss, train_acc, train_top5_acc, val_loss, val_acc, val_top5_acc))

            # Save models
            torch.save(model.state_dict(), "MG_Recog_Weights/final_weight.pkl")

    # Load model and test performance
    else:
        model.load_state_dict(torch.load("MG_Recog_Weights/final_weight.pkl"))
    
    model.eval()
    num_test_correct  = 0
    num_test_examples = 0
    num_top5_correct = 0
    for batch in test_dl:
        x    = batch[0].to(device)
        y    = batch[1].to(device)
        yhat = model(x)
        yhat = yhat.view(len(yhat), 32)

        num_test_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
        num_test_examples += y.shape[0]
        num_top5_correct  += topKCorrect(y, yhat, 5)

    top1_acc = num_test_correct / num_test_examples
    top5_acc = num_top5_correct / num_test_examples
    
    print('Top 1:', round(top1_acc, 4), '| Top 5:', round(top5_acc, 4))
    final_result.write('Top 1: %s\n' % (round(top1_acc, 4)))
    final_result.write('Top 5: %s\n' % (round(top5_acc, 4)))
                        
    # Training End
    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    print('Time total: %5.2f sec' % (total_time_sec))
        
    final_result.close()
    return top1_acc, top5_acc