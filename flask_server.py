import os
import pickle

import jsonpickle
from flask import Flask, request

from feature_set import generate_features_from_text
from log_linear_model import LogLinearModel
from tokenizer import Tokenizer

# define and load model
tokenizer = Tokenizer("data/stopwords_eng.txt")

with open('data/log_linear.model', "rb") as fb:
    model: LogLinearModel = pickle.load(fb)

# define and start web server
app = Flask(__name__)


# response class
class SentimentResponse:
    def __init__(self, predictions, weights, features):
        self.detailed_probabilities = {}

        for prediction in predictions:
            self.detailed_probabilities[prediction[1]] = float(prediction[0])

        if float(predictions[0][0]) == float(predictions[1][0]):
            self.predicted_class = "neutral"
        else:
            self.predicted_class = predictions[0][1]

        self.word_weights = {}
        for feature in features:
            sum_weights: float = 0.0
            sum_weights += float(weights[feature + "-positive"])
            sum_weights -= float(weights[feature + "-negative"])
            self.word_weights[feature] = sum_weights


@app.route('/analyze')
def analyze():
    text: "" = request.args.get('text')
    feature_set = generate_features_from_text(tokenizer, text)
    prediction = model.predict(feature_set, True)

    response = SentimentResponse(prediction, model.weights, feature_set.features)

    return jsonpickle.encode(response, unpicklable=False)


port = os.getenv('PORT', 5000)
app.run(host='0.0.0.0', port=port)
