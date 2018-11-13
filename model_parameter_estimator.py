import itertools
import pickle
from threading import Thread

from confusion_matrix import ConfusionMatrix
from log_linear_model import LogLinearModel


class ModelParameterEstimator:
    def __init__(self, file_validation_set):
        self.learning_rate_candidates = [1, 0.1, 0.01, 0.001]
        self.regularization_rate_candidates = [0.1, 0.01, 0.001, 0.0001];
        self.datasets = ()

        self.scored_candidates = []

        with open(file_validation_set, 'rb') as input:
            self.feature_sets = pickle.load(input)

    def candidate_score(self, candidate, score, cf):
        self.scored_candidates.append((candidate, score, cf))

    def get_best_candidate(self):
        self.scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return self.scored_candidates[0]

    def load_data_sets(self):
        """ load data sets """

        with open('data/feature_sets_training.lst', 'rb') as input:
            feature_sets = pickle.load(input)

        with open('data/feature_sets_dev.lst', 'rb') as input:
            feature_sets_test = pickle.load(input)

        with open('data/feature_sets_validation.lst', 'rb') as input:
            feature_sets_validation = pickle.load(input)

        self.datasets = (feature_sets, feature_sets_test, feature_sets_validation)

    def generate_candidates(self):
        input = [
            self.learning_rate_candidates,
            self.regularization_rate_candidates
        ]

        candidates = list(itertools.product(*input))
        return candidates

    def gridsearch(self):
        """ load all generated feature sets and run 4 training/evaluation via ModelEstimationTask in parallel """
        self.load_data_sets()

        candidates = self.generate_candidates()

        start = 0
        for i in range(0, len(candidates), 4):
            sub_list = candidates[start:i]

            proc = []
            for i in range(len(sub_list)):
                candidate = sub_list[i]
                task = ModelEstimationTask(candidate, self.datasets, self)

                p = Thread(target=task.run)
                p.start()
                proc.append(p)

            for p in proc:
                p.join()

            start = i


class ModelEstimationTask:
    def __init__(self, candidate, data_sets, parent: ModelParameterEstimator):
        self.data_sets = data_sets
        self.candidate = candidate
        self.parent = parent

    def run(self):
        feature_sets = self.data_sets[0]
        feature_sets_test = self.data_sets[1]
        feature_sets_validation = self.data_sets[2]

        model = LogLinearModel(["pos", "neg"])

        model.auto_train(feature_sets, feature_sets_test, learning_rate=self.candidate[0],
                         regularization=self.candidate[1])

        cf = ConfusionMatrix(["pos", "neg"])

        for feature_set in feature_sets_validation:
            predicted_class = model.predict(feature_set)[0][1]
            print("Predicted: " + predicted_class + ", Real: " + feature_set.class_name)

            cf.add_prediction(feature_set.class_name, predicted_class)

        self.parent.candidate_score(dict(l2rate=self.candidate[0], lr=self.candidate[1]), cf.accuracy_average(), cf)


mpe = ModelParameterEstimator('data/feature_sets_validation.lst')
mpe.gridsearch()
print(mpe.scored_candidates)
print(mpe.get_best_candidate())
