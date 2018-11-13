import numpy as np


class ConfusionMatrix:
    def __init__(self, class_list, matrix=None):
        self.number_of_classes = len(class_list)
        self.classes = class_list
        if matrix is None:
            self.matrix = []
            self.clear()
        else:
            self.matrix = matrix

    def clear(self):
        self.matrix = np.zeros((self.number_of_classes, self.number_of_classes))

    def add_prediction(self, real_class, predicted_class):
        x_index = self.classes.index(real_class)
        y_index = self.classes.index(predicted_class)

        self.matrix[x_index, y_index] += 1

    def get_error_context_for_class(self, positive):
        class_index = self.classes.index(positive)
        total_predicted = np.sum(self.matrix[class_index, :])
        total_real = np.sum(self.matrix[:, class_index])

        tp = self.matrix[class_index, class_index]
        fp = total_predicted - tp
        fn = total_real - tp
        tn = max(np.matrix(self.matrix).sum() - tp - fn - fp, 0.0)

        return tp, fp, tn, fn

    def precision(self, clazz):
        tp, fp, tn, fn = self.get_error_context_for_class(clazz)

        divisor = (tp + fp)
        if divisor == 0:
            return 0

        return tp / divisor

    def precision_average(self):
        average_precision = 0.0
        for clazz in self.classes:
            average_precision += self.precision(clazz)

        return average_precision / len(self.classes)

    def recall(self, clazz):
        tp, fp, tn, fn = self.get_error_context_for_class(clazz)

        divisor = (tp + fn)
        if divisor == 0:
            return 0

        return tp / divisor

    def recall_average(self):
        average_recall = 0.0
        for clazz in self.classes:
            average_recall += self.recall(clazz)

        return average_recall / len(self.classes)

    def f_measure(self, clazz):
        precision = self.precision(clazz)
        recall = self.recall(clazz)

        return (2 * precision * recall) / (precision + recall)

    def f_measure_average(self):
        average_f_measure = 0.0
        for clazz in self.classes:
            average_f_measure += self.f_measure(clazz)

        return average_f_measure / len(self.classes)

    def accuracy(self, clazz):
        tp, fp, tn, fn = self.get_error_context_for_class(clazz)

        divisor = (tp + tn + fn + fp)
        if divisor == 0:
            return 0

        return (tp + tn) / divisor

    def accuracy_average(self):
        average_accuracy = 0.0
        for clazz in self.classes:
            average_accuracy += self.accuracy(clazz)

        return average_accuracy / len(self.classes)

    def __repr__(self):
        return str(self.matrix)