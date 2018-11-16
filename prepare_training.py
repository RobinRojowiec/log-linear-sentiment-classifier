#!/usr/bin/python3

from feature_set import generate_feature_sets
from tf_idf import normalize_and_store, normalize

# define paths for training and dev data
paths_training = [["data/training/pos", "positive"], ["data/training/neg", "negative"]]
paths_dev = [["data/dev/pos",
              "positive"], ["data/dev/neg", "negative"]]
paths_validation = [["data/validation/pos", "positive"], ["data/validation/neg", "negative"]]

print("Starting processing...")
all_feature_sets = []

feature_sets_training = generate_feature_sets(paths_training, None)
all_feature_sets.extend(feature_sets_training)

feature_sets_dev = generate_feature_sets(paths_dev, None)
all_feature_sets.extend(feature_sets_dev)

feature_sets_validation = generate_feature_sets(paths_validation, None)
all_feature_sets.extend(feature_sets_validation)

# generate all features from training and test data
tf_idf_pos, tf_idf_neg = normalize_and_store(all_feature_sets, "tf_idf_tr_pos.freq", "tf_idf_tr_neg.freq", None)

normalize_and_store(feature_sets_training, None, None, "feature_sets_training.lst")
print("Training data processed!")

normalize_and_store(feature_sets_dev, None, None, "feature_sets_dev.lst")
print("Dev data processed!")

normalize_and_store(feature_sets_validation, None, None, "feature_sets_validation.lst")
print("Validation data processed!")

print("Finished!")