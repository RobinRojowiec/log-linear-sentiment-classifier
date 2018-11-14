import copy
import math
import pickle
import random
import sys
from collections import defaultdict
from decimal import *

from confusion_matrix import ConfusionMatrix


class LogLinearModel:
    def __init__(self, classes):
        self.weights = defaultdict(float)
        self.classes = classes

    def auto_train(self, feature_sets_training, feature_sets_dev, learning_rate=0.1, regularization=0.001,
                   use_log_freq=False,
                   gradient_descent_threshold=0.5):
        """ trains the model automatically, meaning no iteration count is required,
        because the training stops when the model converges (the gradient delta between iterations is very small)"""

        # initialize variables (last_gradient is the latest gradient and at the first run,
        # we want to make sure the gradient delta is bog enough)
        last_gradient = sys.maxsize
        best_acc_weights: dict = self.weights
        acc_highest_value: float = 0
        iteration: int = 1

        while True:
            random.shuffle(feature_sets_training)

            avg_gradient = self.training_iteration(feature_sets_training, learning_rate, regularization, use_log_freq)

            cf: ConfusionMatrix = self.test(feature_sets_dev, use_log_freq)

            iteration += 1
            print(cf)
            print("Iteration " + str(iteration) + ", Average Gradient: " + str(avg_gradient) + ", Accuracy:" + str(
                cf.accuracy_average() * 100.0) + "%")

            # the weights of an iteration which got better results than before are copied
            # they will be saved at after the last iteration and we don't wont to save
            # every time better weights are found
            if cf.accuracy_average() > acc_highest_value:
                best_acc_weights = copy.deepcopy(self.weights)
                acc_highest_value = cf.accuracy_average()

            # if gradient threshold is exceeded, the training stops
            if last_gradient - avg_gradient < gradient_descent_threshold:
                break;
            else:
                last_gradient = avg_gradient

        # set the current weights to the best weights found before saving
        self.weights = best_acc_weights

        with open('data/log_linear.model', "wb") as fb:
            pickle.dump(self, fb, -1)

    def training_iteration(self, feature_sets_training, learning_rate, regulator=None, use_log_freq=False):
        """ Fits the model to the given feature set """

        average_gradient = 0

        for feature_set in feature_sets_training:
            average_gradient += self.fit(feature_set, learning_rate, regulator, use_log_freq)

        return average_gradient / len(feature_sets_training)

    def fit(self, feature_set, learning_rate, regulator=None, use_log_freq=False):
        """ Runs an update cycle by calculating the gradient of the current feature set and
        adding it to the feature weights """

        # get the current prediction of the model
        predictions = self.predict(feature_set, use_log_freq)

        # gets a tuple with the prediction values for both classes
        correct_class = next(cls for cls in predictions if cls[1] == feature_set.class_name)

        # average gradient for learning rate optimization
        avg_gradient = 0

        # update weights
        for feature in feature_set.features:
            feature_freq = feature_set.features[feature]
            feature_name = feature + "-" + correct_class[1]

            avg_gradient += self.update_weight_for_feature(correct_class, feature_name, feature_freq, learning_rate,
                                                           regulator, use_log_freq)

        return avg_gradient

    def update_weight_for_feature(self, correct_class, feature_name, freq, learning_rate, regulate=None,
                                  use_log_freq=False):
        """ Updates the weight for one feature by calculating its gradient"""

        if use_log_freq:
            freq = math.log(freq + 1)

        gradient = Decimal(freq - (freq * float(correct_class[0])))

        if regulate is not None:
            gradient = gradient - Decimal(freq * regulate)

        weight = Decimal(self.weights[feature_name])
        self.weights[feature_name] = weight + (gradient * Decimal(learning_rate))

        return abs(gradient)

    def get_probability(self, feature_set, clazz, use_log_freq):
        """ returns the probability of a feature set for the given class"""
        probability = Decimal()

        for feature in feature_set.features:
            feature_name = feature + "-" + clazz

            freq = feature_set.features[feature]
            if use_log_freq:
                freq = math.log(freq + 1)

            weight = Decimal(self.weights[feature_name])
            probability += weight * Decimal(freq)

        return probability.exp()

    def predict(self, feature_set, use_log_freq=False):
        """ predicts the probabilities for all classes given this feature set"""
        predictions = []
        sum_probabilities = Decimal()

        # collect all probabilities
        for clazz in self.classes:
            prediction = self.get_probability(feature_set, clazz, use_log_freq)
            sum_probabilities += prediction

            predictions.append([prediction, clazz])

        # normalize probabilities
        for prediction in predictions:
            prediction[0] = prediction[0] / sum_probabilities

        predictions.sort(key=lambda x: x[0], reverse=True)
        return predictions

    def test(self, test_sets, use_log_freq=False):
        """ Runs an evaluation using the dev set """
        cf = ConfusionMatrix(self.classes)

        for feature_set_test in test_sets:
            prediction = self.predict(feature_set_test, use_log_freq)
            predicted_class = prediction[0][1]

            cf.add_prediction(feature_set_test.class_name, predicted_class)

        return cf
