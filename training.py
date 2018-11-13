#!/usr/bin/python3
import sys

from feature_set import generate_feature_sets
from log_linear_model import LogLinearModel

# Set parameters for training or take default values
params = sys.argv[1:]

if len(params) > 0:
    learning_rate = float(params[0])
else:
    learning_rate = 0.01

if len(params) > 1:
    regularization = float(params[1])
else:
    regularization = 0.001

# define paths for training and dev data
paths_training = [["data/training/pos", "pos"], ["data/training/neg", "neg"]]
paths_dev = [["data/dev/pos", "pos"], ["data/dev/neg", "neg"]]

print("Starting training")

# generate all features from training and test data
feature_sets_training = generate_feature_sets(paths_training, "feature_sets_training")
feature_sets_dev = generate_feature_sets(paths_dev, "feature_sets_dev")

# training the model (auto here means the trainings stops when a specific gradient threshold has been exceeded)
model = LogLinearModel(["pos", "neg"])
model.auto_train(feature_sets_training, feature_sets_dev, learning_rate, regularization, True)

print("Finished!")
