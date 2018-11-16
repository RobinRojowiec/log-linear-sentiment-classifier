import os
import pickle

import jsonpickle
from flask import Flask, request

from feature_set import generate_features_from_text
from log_linear_model import LogLinearModel
from tokenizer import Tokenizer
from tf_idf import load_tf_idf, normalize

# define and load model
tokenizer = Tokenizer("data/stopwords_eng.txt")

# load linear model
with open('data/log_linear.model', "rb") as fb:
    model: LogLinearModel = pickle.load(fb)

# load tf-idf weights
tf_idf_pos, tf_idf_neg = load_tf_idf("tf_idf_tr_pos.freq", "tf_idf_tr_neg.freq")

# define and start web server
app = Flask(__name__)


# response class
class SentimentResponse:
    def __init__(self, text, predictions, weights):
        # set the probabilities for each predicted class
        self.detailed_probabilities = {}
        for prediction in predictions:
            self.detailed_probabilities[prediction[1]] = float(prediction[0])

        # find the best class
        if float(predictions[0][0]) == float(predictions[1][0]):
            self.predicted_class = "neutral"
        else:
            self.predicted_class = predictions[0][1]

        # for visualization, calculate the tendency of each token for class as the delta between both weights
        bags = tokenizer.create_bow_per_token(text)
        self.word_weights = []
        for bag in bags:
            self.word_weights.append([bag[0], self.calculate_tendency(bag, weights)])

    def calculate_tendency(self, token_bag, weights):
        features = token_bag[1]
        sum_weights: float = 0.0
        for feature in features:
            sum_weights += float(weights[feature + "-" + model.classes[0]])
            print(weights[feature + "-" + model.classes[0]])
            sum_weights -= float(weights[feature + "-" + model.classes[1]])
            print(weights[feature + "-" + model.classes[0]])

        return sum_weights


@app.route('/analyze')
def analyze():
    text: "" = request.args.get('text')
    feature_set = generate_features_from_text(tokenizer, text)
    normalize([feature_set], tf_idf_pos, tf_idf_neg, None)

    prediction = model.predict(feature_set, True)

    response = SentimentResponse(text, prediction, model.weights)

    return jsonpickle.encode(response, unpicklable=False)


port = os.getenv('PORT', 5000)
app.run(host='0.0.0.0', port=port)
