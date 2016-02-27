from __future__ import division
from collections import Counter, defaultdict
from heapq import nlargest
from math import log, exp
import random


class Classifier(object):
    def __init__(self):
        pass

    def fit(self, examples, outputs):
        pass

    def predict(self, example):
        pass


class NaiveBayes(Classifier):
    def __init__(self):
        self.penalty = 18
        self.class_prob = defaultdict(float)
        self.feature_prob = defaultdict(lambda: defaultdict(float))
        self.all_features = set()

    def fit(self, examples, outputs):
        example_count = len(examples)
        assert example_count == len(outputs), "input/output size mismatch"

        class_count = Counter(outputs)
        for category, count in class_count.items():
            self.class_prob[category] = count / example_count

        for example, category in zip(examples, outputs):
            self.all_features.update(example)
            for feature in example:
                self.feature_prob[category][feature] += 1

        for category in self.feature_prob:
            class_word_count = sum(self.feature_prob[category].values())
            class_word_count += len(self.all_features)
            for feature in self.feature_prob[category]:
                self.feature_prob[category][feature] += 1
                self.feature_prob[category][feature] /= class_word_count

    def predict(self, x_data):
        class_score = defaultdict(float)

        for category in self.class_prob:
            prior = self.class_prob[category]
            class_score[category] = log(prior)
            for feature in x_data:
                if feature in self.feature_prob[category]:
                    cp = self.feature_prob[category][feature]
                    class_score[category] += log(cp)
                elif feature in self.all_features:
                    class_score[category] -= self.penalty

        return max(class_score, key=lambda x: class_score[x])


class DecisionStump(Classifier):
    def __init__(self):
        self.ratio = 0.88

    def fit(self, examples, outputs, weights=None):
        assert len(examples) == len(outputs), "input/output size mismatch"
        self.classes = list(set(outputs))

        weights = weights or [1 for i in examples]

        class_count = defaultdict(float)
        feature_count = defaultdict(lambda: defaultdict(float))
        for example, category, w in zip(examples, outputs, weights):
            class_count[category] += w
            for feature in example:
                feature_count[feature][category] += w

        entropy = 0
        w_sum = sum(weights)
        for count in class_count.values():
            ratio = count / w_sum
            entropy -= ratio * log(ratio, 2)

        IG = defaultdict(float)
        for feature in feature_count:
            branch_count = sum(feature_count[feature].values())

            true_entropy = 0
            for category in feature_count[feature]:
                ratio = feature_count[feature][category] / branch_count
                true_entropy += ratio * log(ratio, 2)

            false_entropy = 0
            for category in feature_count[feature]:
                false_total = w_sum - branch_count
                true_count = feature_count[feature][category]
                false_count = class_count[category] - true_count
                ratio = false_count / false_total
                if ratio > 0:
                    false_entropy += ratio * log(ratio, 2)

            IG[feature] = entropy
            IG[feature] += true_entropy * branch_count / w_sum
            IG[feature] += false_entropy * (1 - branch_count / w_sum)

        fnum = int(self.ratio * len(feature_count))

        self.parameters = defaultdict(lambda: defaultdict(float))
        for feature in nlargest(fnum, IG, key=IG.get):
            ratio = IG[feature] / sum(feature_count[feature].values())
            for category, value in feature_count[feature].items():
                self.parameters[feature][category] = value * ratio

    def predict(self, x_data):
        class_score = defaultdict(float)

        for feature in x_data:
            for category in self.parameters[feature]:
                score = self.parameters[feature][category]
                class_score[category] += score

        if class_score:
            return max(class_score, key=class_score.get)
        else:
            return random.choice(self.classes)


class AdaBoost_SAMME(Classifier):
    def __init__(self, n_iter=1, weak_learner=DecisionStump):
        self.Classifier = weak_learner
        self.n_iter = n_iter

    def fit(self, examples, outputs):
        example_count = len(examples)
        self.classes = set(outputs)
        self.K = len(self.classes)

        self.errors = []
        self.alphas = []
        self.classifiers = []

        weights = [1/example_count for i in range(example_count)]

        for n in range(self.n_iter):
            cls = self.Classifier()
            cls.fit(examples, outputs, weights=weights)
            predicted = map(cls.predict, examples)
            self.classifiers.append(cls)

            data = zip(predicted, outputs, weights)
            err = sum(w for a, b, w in data if a != b)
            self.errors.append(err)

            alpha = log((1 - err) / err) + log(self.K - 1)
            self.alphas.append(alpha)

            for i in range(example_count):
                if predicted[i] != outputs[i]:
                    weights[i] *= exp(alpha)

            # Normalize weights
            w_sum = sum(weights)
            weights[:] = [w / w_sum for w in weights]

    def predict(self, x_data):
        class_score = defaultdict(float)

        for cls, alpha in zip(self.classifiers, self.alphas):
            predicted = cls.predict(x_data)
            class_score[predicted] += alpha

        return max(class_score, key=lambda x: class_score[x])
