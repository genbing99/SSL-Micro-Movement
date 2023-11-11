import argparse
from distutils.util import strtobool

from MG_Recog.prepare_data import *
from MG_Recog.train import *

# Follow the experiment settings in paper
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr_features", type=float, default=0.00001)
parser.add_argument("--lr_classifier", type=float, default=0.0001)
parser.add_argument("--pretext_epoch", type=int, default=30)
parser.add_argument("--train_model", type=strtobool, default=False)
opt = parser.parse_args()

epochs = opt.epochs
batch_size = opt.batch_size
lr_features = opt.lr_features
lr_classifier = opt.lr_classifier
pretext_epoch = opt.pretext_epoch
train_model = opt.train_model

# Load data from files
final_subjects, final_dataset, final_emotions, final_videos = load_data()

# Prepare training and test sets
X, y, groupsLabel = prepare_dataset(final_subjects, final_dataset, final_emotions, final_videos)

# Follow iMiGUE benchmark work
train_subject = [2, 4, 5, 6, 9, 10, 12, 13, 14, 15, 17, 18, 20, 21, 22, 23, 25, 28, 29, 30, 31, 35, 36, 38, 39, 41, 42, 43, 44, 46, 47, 48, 52, 53, 59, 61, 62]
test_subject = [1, 3, 7, 8, 11, 16, 19, 24, 26, 27, 32, 33, 34, 37, 40, 45, 49, 50, 51, 54, 55, 56, 57, 58, 60, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72]

# Train or test model
top1_acc, top5_acc = train(X, y, groupsLabel, train_subject, test_subject, epochs=epochs, lr_features=lr_features, lr_classifier=lr_classifier, batch_size=batch_size, pretext_epoch=pretext_epoch, train_model=train_model)
