import cv2
import time
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader
from numpy import argmax
from sklearn.model_selection import LeaveOneGroupOut

from ME_Recog.network import *
from ME_Recog.dataloader import *
from ME_Recog.evaluation import *

def train(X, Y, final_subjects, groupsLabel, epochs=200, lr_features=0.000005, lr_classifier=0.00005, batch_size=256, pretext_epoch=10, train_model=False):
    # Create model directory
    os.makedirs("ME_Recog_Weights", exist_ok=True)
        
    # For LOSO
    loso = LeaveOneGroupOut()
    recog_train_index = []
    recog_test_index = []

    # For evaluation
    cur_gt = []
    cur_pred = []

    # Get best score
    all_pred = []
    all_gt = []

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
              
    loss_fn = nn.CrossEntropyLoss()     

    for train_index, test_index in loso.split(X, Y, groupsLabel): # Recognition Leave One Subject Out
        recog_train_index.append(train_index)
        recog_test_index.append(test_index)

    # Resize images to 56x56
    for i in range(len(X)):
        X[i] = cv2.resize(X[i], (56,56))
        X[i] = X[i].astype(np.float32)

    start_time_sec = time.time()
        
    # Write final result to final_result.txt
    final_result = open("ME_Recog_Weights/final_result.txt", "w")

    for subject_count in range(len(final_subjects)): 
        
        model = LightWeight_Network_Recog(pretext_epoch = pretext_epoch).to(device)
        optimizer = torch.optim.Adam([
            {'params': model.features.parameters(), 'lr': lr_features, 'momentum': 0.9},
            {'params': model.classifier.parameters(), 'lr': lr_classifier, 'momentum': 0.9}
        ])

        print('Index: ' + str(subject_count) + ' | Subject : ' + str(final_subjects[subject_count]))
        X_train, X_test = [X[i] for i in recog_train_index[subject_count]], [X[i] for i in recog_test_index[subject_count]] 
        y_train, y_test = [Y[i] for i in recog_train_index[subject_count]], [Y[i] for i in recog_test_index[subject_count]] 

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
            batch_size=256,
            shuffle=False,
        )

        if train_model:

            for epoch in range(1, epochs+1):

                # Training
                model.train()
                train_loss         = 0.0
                num_train_correct  = 0
                num_train_examples = 0
                for batch in train_dl:
                    x    = batch[0].to(device)
                    y    = batch[1].to(device)
                    optimizer.zero_grad()
                    yhat = model(x)
                    yhat = yhat.view(len(yhat), 3)

                    loss = loss_fn(yhat, y)
                    loss.backward()
                    optimizer.step()

                    train_loss         += loss.data.item() * x.size(0)
                    num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_train_examples += x.shape[0]

                train_acc   = num_train_correct / num_train_examples
                train_loss  = train_loss / len(train_dl.dataset)

                # Testing
                model.eval()
                val_loss       = 0.0
                num_val_correct  = 0
                num_val_examples = 0
                for batch in test_dl:
                    x    = batch[0].to(device)
                    y    = batch[1].to(device)
                    yhat = model(x)
                    yhat = yhat.view(len(yhat), 3)
                    loss = loss_fn(yhat, y)

                    val_loss         += loss.data.item() * x.size(0)
                    num_val_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_val_examples += y.shape[0]

                val_acc  = num_val_correct / num_val_examples
                val_loss = val_loss / len(test_dl.dataset)

                if epoch == 1 or epoch % 20 == 0: # Note that val = test herein
                    print('Epoch %3d/%3d, train loss: %5.4f, train acc: %5.4f, val loss: %5.4f, val acc: %5.4f' % (epoch, epochs, train_loss, train_acc, val_loss, val_acc))

                # Save models
                torch.save(model.state_dict(), os.path.join("ME_Recog_Weights/subject_%s.pkl" % (str(final_subjects[subject_count]))))

        # Load model and test performance
        else: 
            model.load_state_dict(torch.load("ME_Recog_Weights/subject_%s.pkl" % (str(final_subjects[subject_count]))))
            model.eval()
            for batch in test_dl:
                x    = batch[0].to(device)
                y    = batch[1].to(device)
                yhat = model(x)
                yhat = yhat.view(len(yhat), 3)

        # For UF1 and UAR computation
        pred  = torch.max(yhat, 1)[1].tolist()
        gt = y.tolist()
        print('Predicted    :', pred)
        print('Ground Truth :', gt)
        print('Evaluation until this subject: ')
        cur_gt.extend(gt)
        cur_pred.extend(pred)
        all_gt.append(gt)
        all_pred.append(pred)
        UF1, UAR = recognition_evaluation(cur_gt, cur_pred, show=False)
        print('')

        final_result.write('Index: %s | Subject: %s\n' % (str(subject_count), str(final_subjects[subject_count])))
        final_result.write('Predicted    : %s\n' % (pred))
        final_result.write('Ground Truth : %s\n' % (gt))
        final_result.write('UF1: %s | UAR: %s\n' % (round(UF1, 4), round(UAR, 4)))
                        
    # Training End
    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    print('Time total:     %5.2f sec' % (total_time_sec))
        
    final_result.close()
    return all_gt, all_pred