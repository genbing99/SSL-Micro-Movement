import argparse
from distutils.util import strtobool

from ME_Spot.prepare_data import *
from ME_Spot.evaluation import *
from ME_Spot.train import *

# Follow the experiment settings in paper
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='CASME_sq') # Either 'CASME_sq' or 'SAMMLV' only
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr_features", type=float, default=0.0005)
parser.add_argument("--lr_classifier", type=float, default=0.001)
parser.add_argument("--pretext_epoch", type=int, default=100)
parser.add_argument("--train_model", type=strtobool, default=False)
opt = parser.parse_args()

dataset_name = opt.dataset_name
epochs = opt.epochs
batch_size = opt.batch_size
lr_features = opt.lr_features
lr_classifier = opt.lr_classifier
pretext_epoch = opt.pretext_epoch
train_model = opt.train_model

# Load data from files
dataset, subjects, subjectsVideos = load_data(dataset_name)

# Load excel
codeFinal = load_excel(dataset_name)

# Prepare label
final_subjects, final_videos, final_samples = load_ground_truth(dataset_name, subjects, subjectsVideos, codeFinal)

# Pseudo-label
pseudo_y, k = pseudo_label(dataset_name, final_samples, dataset)

# Prepare training and test sets
X, Y, groupsLabel = prepare_dataset(dataset, pseudo_y, final_samples)

# Train or test model
total_gt, metric_fn = train(X, Y, groupsLabel, dataset, k, dataset_name, final_subjects, final_samples, epochs=epochs, lr_features=lr_features, lr_classifier=lr_classifier, batch_size=batch_size, pretext_epoch=pretext_epoch, train_model=train_model)
