import os
import pickle

import jsonpickle
from flask import Flask, request

from feature_set import generate_features_from_text
from log_linear_model import LogLinearModel
from sentiment_response import SentimentResponse
from tf_idf import load_tf_idf, normalize
from tokenizer import Tokenizer

# define and load model
tokenizer = Tokenizer("data/stopwords_eng.txt")

# load linear model
with open('data/log_linear.model', "rb") as fb:
    model: LogLinearModel = pickle.load(fb)

# load tf-idf weights
tf_idf_pos, tf_idf_neg = load_tf_idf("tf_idf_tr_pos.freq", "tf_idf_tr_neg.freq")

# define and start web server
app = Flask(__name__)


@app.route('/analyze')
def analyze():
    text: "" = request.args.get('text')
    feature_set = generate_features_from_text(tokenizer, text)
    normalize([feature_set], tf_idf_pos, tf_idf_neg, None)

    prediction = model.predict(feature_set, True)

    response = SentimentResponse(text, tokenizer, prediction, model)

    return jsonpickle.encode(response, unpicklable=False)


port = os.getenv('PORT', 5000)
app.run(host='0.0.0.0', port=port)
