#!/usr/bin/python3
from feature_set import generate_feature_sets

# define paths for training and dev data
paths_training = [["data/training/pos", "positive"], ["data/training/neg", "negative"]]
paths_dev = [["data/dev/pos",
              "positive"], ["data/dev/neg", "negative"]]
paths_validation = [["data/validation/pos", "positive"], ["data/validation/neg", "negative"]]

print("Starting processing...")

# generate all features from training and test data
feature_sets_training = generate_feature_sets(paths_training, "feature_sets_training")
print("Training data processed!")

feature_sets_dev = generate_feature_sets(paths_dev, "feature_sets_dev")
print("Dev data processed!")

feature_sets_validation = generate_feature_sets(paths_validation, "feature_sets_validation")
print("Validation data processed!")

print("Training- and Dev-data processed!")
