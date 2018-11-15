import os
import pickle

from tokenizer import Tokenizer


class FeatureSet:
    def __init__(self, class_name):
        """ A feature set is a collection of features from a file with its corresponding class """
        self.features = {}
        self.class_name = class_name

    def count_feature(self, feature_name):
        if feature_name not in self.features:
            self.features[feature_name] = 1
        else:
            self.features[feature_name] += 1


def generate_features_from_text(tokenizer, text):
    """ generates features from input text """
    bow = tokenizer.create_bag_of_words(text.split())

    feature_set = FeatureSet(None)
    for token in bow:
        feature_set.count_feature(token)

    return feature_set


def generate_feature_sets(paths, file_to_store):
    """ generates a feature set per file in the given paths """
    tokenizer = Tokenizer("data/stopwords_eng.txt")
    feature_sets = []

    for path in paths:
        class_name = path[1]

        for file in os.listdir(path[0]):
            file_path = path[0] + "/" + file
            bow = tokenizer.create_bow_from_file(file_path)

            feature_set = FeatureSet(class_name)
            for token in bow:
                feature_set.count_feature(token)

            feature_sets.append(feature_set)

    if file_to_store is not None:
        with open('data/'+file_to_store+'.lst', 'wb') as output:
            pickle.dump(feature_sets, output, -1)

    return feature_sets
