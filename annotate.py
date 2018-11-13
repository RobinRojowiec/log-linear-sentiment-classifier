from feature_set import FeatureSet, generate_features_from_text
from tokenizer import Tokenizer
import sys
import pickle

text = sys.argv[1]

tokenizer = Tokenizer("data/stopwords_eng.txt")
feature_set = generate_features_from_text(tokenizer, text)

with open('data/log_linear.model', "rb") as fb:
    model = pickle.load(fb)

print(model.predict(feature_set))

