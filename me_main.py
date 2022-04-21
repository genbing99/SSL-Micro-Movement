
import argparse
from distutils.util import strtobool

from ME_Recog.prepare_data import *
from ME_Recog.evaluation import *
from ME_Recog.train import *

# Follow the experiment settings in paper
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--lr_features", type=float, default=0.000001)
parser.add_argument("--lr_classifier", type=float, default=0.00005)
parser.add_argument("--pretext_epoch", type=int, default=100)
parser.add_argument("--train_model", type=strtobool, default=False)
opt = parser.parse_args()

epochs = opt.epochs
batch_size = opt.batch_size
lr_features = opt.lr_features
lr_classifier = opt.lr_classifier
pretext_epoch = opt.pretext_epoch
train_model = opt.train_model

# Load data from files
final_subjects, final_dataset, final_emotions, final_samples = load_data()

# Prepare training and test sets
X, y, groupsLabel = prepare_dataset(final_dataset, final_emotions, final_samples)

# Train or test model
all_gt, all_pred = train(X, y, final_subjects, groupsLabel, epochs=epochs, lr_features=lr_features, lr_classifier=lr_classifier, batch_size=batch_size, pretext_epoch=pretext_epoch, train_model=train_model)

# Final evaluation
subject_evaluation(all_gt, all_pred)
