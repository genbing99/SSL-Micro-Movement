import os
import time
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from MicroMovement.dataloader import *
from MicroMovement.network import *

start_time = time.time()

def train(train_dataset_path, val_dataset_path, train_batch_size, val_batch_size, lr, n_epochs):

    # Create path to save trained models
    os.makedirs("MicroMovement_Weights", exist_ok=True)
    # Write val result
    result = open("MicroMovement_Weights/val_result.txt", "w")

    # Dataset loader
    dataloader = DataLoader(
        OFFDataset(train_dataset_path),
        batch_size=train_batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        OFFDataset(val_dataset_path, mode="val"),
        batch_size=val_batch_size,
        shuffle=True,
    )

    # Initialize network
    model = LightWeight_Network(num_classes=3).cuda()
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Loss Function
    loss_fn = nn.CrossEntropyLoss() 
    # Set gpu/cpu
    cuda = True if torch.cuda.is_available() else False
    tensor_float = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    tensor_long = torch.cuda.LongTensor if cuda else torch.LongTensor
    # Train and val dataset size
    print('Train Batch:', len(dataloader), '| Val Batch:', len(val_dataloader))

    for epoch in range(n_epochs):
        # ----------
        #  Training
        # ----------
        model.train()

        for i, (x, y) in enumerate(dataloader):

            # Configure input
            x = Variable(x.type(tensor_float))
            y = Variable(y.type(tensor_long))

            optimizer.zero_grad()
            
            # Predict label
            yhat = model(x)
            
            # Loss
            train_loss = loss_fn(yhat, y)
            train_loss.backward()
            optimizer.step()

            num_train_correct  = (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples = x.shape[0]
            train_acc   = num_train_correct / num_train_examples
            
            if((i+1) % 10 == 0 or i+1 == len(dataloader)):
                print('Epoch %3d/%3d, Batch %d/%d Loss: %5.4f Acc: %5.4f' % (epoch+1, n_epochs, i+1, len(dataloader), train_loss, train_acc))

        # Save model every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), "MicroMovement_Weights/epoch-%d.pkl" % (epoch + 1))

        # ----------
        # Validation
        # ----------
        model.eval()
        val_loss = 0.0
        num_val_correct  = 0
        num_val_examples = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(val_dataloader):

                # Configure input
                x = Variable(x.type(tensor_float))
                y = Variable(y.type(tensor_long), requires_grad=False)

                # Predict label
                yhat = model(x)
                
                # Loss
                val_loss = loss_fn(yhat, y)    
                num_val_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
                num_val_examples += x.shape[0]

            val_acc   = num_val_correct / num_val_examples

            print('--- Val Epoch %3d/%3d, Val Loss: %5.4f Val Acc: %5.4f ---' % (epoch+1, n_epochs, val_loss, val_acc))
        result.write('Val Epoch %3d/%3d | Val Loss: %5.4f | Val Acc: %5.4f\n' % (epoch+1, n_epochs, val_loss, val_acc))

    print('Total time taken: ', round(time.time() - start_time), 3)
    result.close()
