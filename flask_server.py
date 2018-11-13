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
class SentimentReponse:
    def __init__(self, best_class, predictions):
        self.predicted_class = best_class
        self.detailed_probabilities = {}

        for prediction in predictions:
            self.detailed_probabilities[prediction[1]] = float(prediction[0])


@app.route('/analyze')
def analyze():
    text: "" = request.args.get('text')
    feature_set = generate_features_from_text(tokenizer, text)
    prediction = model.predict(feature_set)

    response = SentimentReponse(prediction[0][1], prediction)

    return jsonpickle.encode(response, unpicklable=False)


port = os.getenv('PORT', 5000)
app.run(host='0.0.0.0', port=port)
