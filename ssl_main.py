import argparse
from distutils.util import strtobool

from MicroMovement.train import *

# Follow the experiment settings in paper
parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", type=int, default=256)
parser.add_argument("--val_batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.00005)

# Specify path for SSL training dataset, we use CASME_sq in the paper
parser.add_argument("--train_dataset_path", type=str, default='CASME_sq/OFF_images/training')
parser.add_argument("--val_dataset_path", type=str, default='CASME_sq/OFF_images/validation')

opt = parser.parse_args()

train_batch_size = opt.train_batch_size
val_batch_size = opt.val_batch_size
n_epochs = opt.n_epochs
lr = opt.lr
train_dataset_path = opt.train_dataset_path
val_dataset_path = opt.val_dataset_path

# Train the MicroMovement SSL
train(train_dataset_path, val_dataset_path, train_batch_size, val_batch_size, lr, n_epochs)