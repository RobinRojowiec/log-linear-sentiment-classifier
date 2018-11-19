import math
import pickle
from collections import Counter
from collections import defaultdict

from feature_set import FeatureSet


class TFIDF:
    def __init__(self, class_name: ""):
        self.doc_freq = defaultdict(int)
        self.class_name = class_name
        self.doc_count = 0

    def add_counts(self, feature_set: FeatureSet):
        for token in feature_set.features:
            doc_count: int = self.doc_freq[token]
            doc_count += 1
            self.doc_freq[token] = doc_count

        self.doc_count += 1

    def normalize(self, feature_set: FeatureSet):
        for feature in feature_set.features:
            normalized: float = 1.0  # self.get_tf(feature, feature_set) * self.get_idf(feature) disabled
            feature_set.features[feature] = normalized

    def get_idf(self, token):
        token_doc_freq = self.doc_freq[token]
        if token_doc_freq == 0:
            return 0
        return math.log(self.doc_count/token_doc_freq)

    def get_tf(self, token, feature_set: FeatureSet):
        freq_dict = Counter(feature_set.features)
        max_freq: float = float(freq_dict.most_common(1)[0][1])
        token_freq = freq_dict[token]
        return token_freq/max_freq


def normalize(feature_sets: [], tf_idf_pos: TFIDF, tf_idf_neg: TFIDF, file_to_store: ""):
    for feature_set in feature_sets:
        if feature_set.class_name == "positive":
            tf_idf_pos.normalize(feature_set)
        else:
            tf_idf_neg.normalize(feature_set)

    if file_to_store:
        with open('data/'+file_to_store, 'wb') as feature_sets_file:
            pickle.dump(feature_sets, feature_sets_file)


def normalize_and_store(feature_sets: [], pos_file = None, neg_file = None, file_to_store = None):
    """Does a local TF-IDF Normalization on the feature_sets"""
    tf_idf_pos = TFIDF("positive")
    tf_idf_neg = TFIDF("negative")

    for feature_set in feature_sets:
        if feature_set.class_name == "positive":
            tf_idf_pos.add_counts(feature_set)
        else:
            tf_idf_neg.add_counts(feature_set)

    normalize(feature_sets,tf_idf_pos, tf_idf_neg, file_to_store)

    if pos_file:
        with open('data/'+pos_file, 'wb') as tf_idf_pos_file:
            pickle.dump(tf_idf_pos, tf_idf_pos_file)

    if neg_file:
        with open('data/'+neg_file, 'wb') as tf_idf_neg_file:
            pickle.dump(tf_idf_neg, tf_idf_neg_file)

    return tf_idf_pos, tf_idf_neg


def load_tf_idf(pos_file: "", neg_file: ""):
    with open('data/' + pos_file, 'rb') as tf_idf_pos_file:
        tf_idf_pos: TFIDF = pickle.load(tf_idf_pos_file)

    with open('data/' + neg_file, 'rb') as tf_idf_neg_file:
        tf_idf_neg: TFIDF = pickle.load(tf_idf_neg_file)

    return tf_idf_pos, tf_idf_neg
