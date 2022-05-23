import time
import torch
import os
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import DataLoader
from numpy import argmax
import random
import cv2
from skimage.util import random_noise
from sklearn.model_selection import LeaveOneGroupOut
from collections import Counter
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d

from ME_Spot.network import *
from ME_Spot.dataloader import *
from ME_Spot.evaluation import *

def shuffling(X, y):
    shuf = list(zip(X, y))
    random.shuffle(shuf)
    X, y = zip(*shuf)
    return list(X), list(y)
    
def data_augmentation(X, y):
    transformations = {
        0: lambda image: np.fliplr(image), 
        1: lambda image: cv2.GaussianBlur(image, (7,7), 0),
        2: lambda image: random_noise(image),
    }
    y1=y.copy()
    for index, label in enumerate(y1):
        if (label==1): #Only augment on expression samples (label=1)
            for augment_type in range(3):
                img_transformed = transformations[augment_type](X[index]).reshape(56,56,3)
                X.append(np.array(img_transformed))
                y.append(1)
    return X, y

def train(X, Y, groupsLabel, dataset, k, dataset_name, final_subjects, final_samples, epochs, lr_features, lr_classifier, batch_size, pretext_epoch, train_model):

    # Create model directory
    os.makedirs("ME_Spot_Weights_" + dataset_name, exist_ok=True)
    
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    loss_fn = nn.MSELoss()

    # Write final result to final_result.txt
    final_result = open("ME_Spot_Weights_" + dataset_name + "/final_result.txt", "w")
    
    start = time.time()
    logo = LeaveOneGroupOut()
    logo.get_n_splits(X, Y, groupsLabel)
    epochs = epochs
    batch_size = batch_size
    subject_count = 0
    total_gt = 0
    metric_fn = MeanAveragePrecision2d(num_classes=1)
    p = 0.55 #From our analysis, 0.55 achieved the highest F1-Score

    for train_index, test_index in logo.split(X, Y, groupsLabel): # Leave One Subject Out
        subject_count+=1
        cur_gt = 0
        for i in final_samples[subject_count-1]:
            total_gt += len(i)
            cur_gt += len(i)

        print('Subject : ' + str(subject_count))

        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index] #Get training set
        Y_train, Y_test = [Y[i] for i in train_index], [Y[i] for i in test_index] #Get testing set

        if train_model:
            #Downsampling non expression samples the dataset by 1/2 to reduce dataset bias 
            print('Dataset Labels', Counter(Y_train))
            unique, uni_count = np.unique(Y_train, return_counts=True) 
            rem_count = int(uni_count.max()*1/2)

            #Randomly remove non expression samples (With label 0) from dataset
            rem_index = random.sample([index for index, i in enumerate(Y_train) if i==0], rem_count) 
            rem_index += (index for index, i in enumerate(Y_train) if i>0)
            rem_index.sort()
            X_train = [X_train[i] for i in rem_index]
            Y_train = [Y_train[i] for i in rem_index]
            print('After Downsampling Dataset Labels', Counter(Y_train))

            #Data augmentation to the micro-expression samples only
            X_train, Y_train = data_augmentation(X_train, Y_train)
            print('After Augmentation Dataset Labels', Counter(Y_train))

            #Shuffle the training set
            X_train, Y_train = shuffling(X_train, Y_train)
            print('Done Shuffling')
            
        # Initialize training dataloader
        X_train = torch.Tensor(np.array(X_train)).permute(0,3,1,2)
        Y_train = torch.Tensor(np.array(Y_train))
        train_dl = DataLoader(
            OFFSpottingDataset((X_train, Y_train)),
            batch_size=batch_size,
            shuffle=True,
        )

        # Initialize testing dataloader
        X_test = torch.Tensor(np.array(X_test)).permute(0,3,1,2)
        Y_test = torch.Tensor(np.array(Y_test))
        test_dl = DataLoader(
            OFFSpottingDataset((X_test, Y_test)),
            batch_size=batch_size,
            shuffle=False,
        )
        
        # print('------Initializing Network-------') #To reset the model at every LOSO testing
        
        model = LightWeight_Network_Spot(pretext_epoch = pretext_epoch).to(device)
        optimizer = torch.optim.Adam([
            {'params': model.features.parameters(), 'lr': lr_features, 'momentum': 0.9},
            {'params': model.classifier.parameters(), 'lr': lr_classifier, 'momentum': 0.9}
        ])
        
        if train_model:
            for epoch in range(1, epochs+1):

                # Training
                model.train()
                train_loss = 0.0
                for batch in train_dl:
                    x    = batch[0].to(device)
                    y    = batch[1].to(device)
                    optimizer.zero_grad()
                    yhat = model(x).view(-1)
                    loss = loss_fn(yhat, y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.data.item()

                train_loss  = train_loss / len(train_dl)

                if epoch == 1 or epoch % 1 == 0: # Note that val = test herein
                    
                    # Testing
                    model.eval()
                    val_loss = 0.0
                    result_all = np.array([])
                    for batch in test_dl:
                        x    = batch[0].to(device)
                        y    = batch[1].to(device)
                        yhat = model(x).view(-1)
                        result = yhat.cpu().data.numpy()
                        result_all = np.append(result_all, result)
                        loss = loss_fn(yhat, y)
                        val_loss += loss.data.item()

                    val_loss = val_loss / len(test_dl)
                    preds, gt, metric = spotting(dataset, final_samples, k, result_all, subject_count, p, show=False)
                    TP, FP, FN, F1_score, precision, recall = evaluation(cur_gt, metric)
                    print('Epoch %3d/%3d, train loss: %5.4f, val loss: %5.4f, TP:%d FP:%d FN:%d' % (epoch, epochs, train_loss, val_loss, TP, FP, FN))

                # Save models
                torch.save(model.state_dict(), os.path.join("ME_Spot_Weights_%s/subject_%s.pkl" % (dataset_name, str(final_subjects[subject_count-1]))))
            
        # For model testing
        else:
            model.load_state_dict(torch.load("ME_Spot_Weights_%s/subject_%s.pkl" % (dataset_name, str(final_subjects[subject_count-1]))))
            model.eval()
            val_loss = 0.0
            result_all = np.array([])
            for batch in test_dl:
                x    = batch[0].to(device)
                y    = batch[1].to(device)
                yhat = model(x).view(-1)
                result = yhat.cpu().data.numpy()
                result_all = np.append(result_all, result)
        
        preds_all, gt_all, metric = spotting(dataset, final_samples, k, result_all, subject_count, p, show=False) # Can change to True to visualize the results

        for i in range(len(preds_all)):
            metric_fn.add(np.array(preds_all[i]), np.array(gt_all[i])) #IoU = 0.5 according to MEGC2020 metrics
        TP, FP, FN, F1_score, precision, recall = evaluation(total_gt, metric_fn)
        print('Cumulative result: TP:%d FP:%d FN:%d F1_score:%5.4f' % (TP, FP, FN, F1_score))

        final_result.write('Index: %s | Subject: %s\n' % (str(subject_count-1), str(final_subjects[subject_count-1])))
        final_result.write('TP:%d FP:%d FN:%d F1_score:%5.4f\n' % (TP, FP, FN, F1_score))

        print('Done Subject', subject_count)
        
    end = time.time()
    print('Total time taken for training & testing: ' + str(end-start) + 's')
    final_result.close()

    return total_gt, metric_fn