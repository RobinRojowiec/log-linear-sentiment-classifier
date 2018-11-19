from log_linear_model import LogLinearModel
from tokenizer import Tokenizer


# response class
class SentimentResponse:
    def __init__(self, text: "", tokenizer: Tokenizer, predictions: [], model: LogLinearModel):
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
        self.word_weights = []
        self.calculate_tendency(tokenizer, text, model.classes, model.weights)

        # calculate sentiment per token based on context
        self.context_sentiment = []
        self.calculate_context_sentiment(tokenizer)

    def calculate_tendency(self, tokenizer: Tokenizer, text: "", classes: [], weights):
        bags = tokenizer.create_bow_per_token(text)
        for bag in bags:
            features = bag[1]
            sum_weights: float = 0.0
            for feature in features:
                sum_weights += float(weights[feature + "-" + classes[0]])
                sum_weights -= float(weights[feature + "-" + classes[1]])

            self.word_weights.append([bag[0], sum_weights])

    def calculate_context_sentiment(self, tokenizer: Tokenizer):
        """extracting 2x2 window context and sum up ignoring current word sentiment"""
        for i in range(len(self.word_weights)):
            context = []
            current = self.word_weights[i][0]

            if current.lower() not in tokenizer.stopwords:
                # left context
                if i - 1 >= 0:
                    context.append(self.word_weights[i - 1])
                    if i - 2 >= 0:
                        context.append(self.word_weights[i - 2])

                # right context
                if i + 1 < len(self.word_weights):
                    context.append(self.word_weights[i + 1])
                    if i + 2 < len(self.word_weights):
                        context.append(self.word_weights[i + 2])

                context_sum = sum([z[1] for z in context]) / len(context)
                self.context_sentiment.append((current, context_sum))
