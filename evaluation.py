#!/usr/bin/python3
import pickle
from feature_set import generate_feature_sets
from confusion_matrix import ConfusionMatrix
import csv


print("Starting evaluation")

# define paths and initialize confusion matrix
paths_validation = [["data/validation/pos", "positive"], ["data/validation/neg", "negative"]]
feature_sets_validation = generate_feature_sets(paths_validation, "feature_sets_validation")
cf = ConfusionMatrix(["positive", "negative"])


# load the training weights
with open('data/log_linear.model', "rb") as fb:
    model = pickle.load(fb)


# evaluate against all sets and store results in csv file
with open('data/validation_report.csv', 'w', newline='\n') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
    csv_writer.writerow(["Actual Class", "Predicted Class"])

    for feature_set in feature_sets_validation:
        prediction = model.predict(feature_set)
        predicted_class = prediction[0][1]

        csv_writer.writerow([feature_set.class_name, predicted_class])
        cf.add_prediction(feature_set.class_name, predicted_class)


# print out overall results
print(cf)
print("Accuracy: "+str(cf.accuracy_average()))
print("Recall: "+str(cf.recall_average()))
print("Precision: "+str(cf.precision_average()))
print("F-Measure: "+str(cf.f_measure_average()))
