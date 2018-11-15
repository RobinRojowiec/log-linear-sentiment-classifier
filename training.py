#!/usr/bin/python3
import pickle
import sys

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
paths_training = [["data/training/pos", "positive"], ["data/training/neg", "negative"]]
paths_dev = [["data/dev/pos", "positive"], ["data/dev/neg", "negative"]]

print("Loading data...")

# generate all features from training and test data

with open("data/feature_sets_training.lst", mode='rb') as training_file:
    feature_sets_training: [] = pickle.load(training_file)

with open("data/feature_sets_dev.lst", mode='rb') as dev_file:
    feature_sets_dev: [] = pickle.load(dev_file)

print("Starting training")

# training the model (auto here means the trainings stops when a specific gradient threshold has been exceeded)
model = LogLinearModel(["positive", "negative"])
model.auto_train(feature_sets_training, feature_sets_dev, learning_rate, regularization, True)

print("Finished!")
