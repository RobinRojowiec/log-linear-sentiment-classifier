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
    def __init__(self, predictions):
        self.detailed_probabilities = {}

        for prediction in predictions:
            formatted_value = "{0:.2f}%".format(float(prediction[0])*100.0)
            self.detailed_probabilities[prediction[1]] = formatted_value

        if float(predictions[0][0]) == float(predictions[1][0]):
            self.predicted_class = "neutral"
        else:
            self.predicted_class = predictions[0][1]


@app.route('/analyze')
def analyze():
    text: "" = request.args.get('text')
    feature_set = generate_features_from_text(tokenizer, text)
    print(feature_set.features)
    prediction = model.predict(feature_set, True)

    response = SentimentResponse(prediction)

    return jsonpickle.encode(response, unpicklable=False)


port = os.getenv('PORT', 5000)
app.run(host='0.0.0.0', port=port)
