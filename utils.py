from __future__ import division
from collections import Counter, defaultdict
import math
import csv


class CrossValidation(object):
    '''
    Iterator that returns 1/k of the data as test data and
    the rest as train data, for every of the k pieces.
    '''
    def __init__(self, examples, outputs, k=10):
        assert len(examples) == len(outputs)

        self.examples = examples
        self.outputs = outputs
        self.k = len(outputs) // k
        self.i = 0

    def __iter__(self):
        return self

    def next(self):
        s, e = self.i * self.k, (self.i + 1) * self.k
        if s >= len(self.examples):
            raise StopIteration
        self.i += 1

        train_data = self.examples[:s] + self.examples[e:]
        train_result = self.outputs[:s] + self.outputs[e:]

        test_data = self.examples[s:e]
        test_result = self.outputs[s:e]

        return train_data, train_result, test_data, test_result


def compute_IG(examples, categories, weights=None):
    '''
    Computes the entropy of each feature
    '''

    weights = weights or [1 for i in examples]

    entropy = 0
    class_count = Counter(categories)
    for count in class_count.values():
        ratio = count / len(categories)
        entropy -= ratio * math.log(ratio, 2)

    all_features = set(f for example in examples for f in example)
    feature_count = defaultdict(lambda: defaultdict(float))
    for example, category, w in zip(examples, categories, weights):
        for feature in example:
            feature_count[feature][category] += w

    IG = defaultdict(float)
    for feature in all_features:
        branch_count = sum(feature_count[feature].values())

        true_entropy = 0
        for category in feature_count[feature]:
            ratio = feature_count[feature][category] / branch_count
            true_entropy += ratio * math.log(ratio, 2)

        false_entropy = 0
        for category in feature_count[feature]:
            false_total = len(examples) - branch_count
            true_count = feature_count[feature][category]
            false_count = class_count[category] - true_count
            ratio = false_count / false_total
            if ratio > 0:
                false_entropy += ratio * math.log(ratio, 2)

        IG[feature] = entropy
        IG[feature] += true_entropy * branch_count / len(examples)
        IG[feature] += false_entropy * (1 - branch_count / len(examples))

    return IG


def print_CC(actual, predicted):
    '''
    Prints out the confusion matrix
    '''

    CC = defaultdict(lambda: defaultdict(float))
    for a, b in zip(actual, predicted):
        CC[a][b] += 1

    all_classes = list(set(actual + predicted))
    print '        |' + ' '.join('%8s' % c for c in all_classes)
    print '-' * 8 * (len(all_classes) + 2)
    for c1 in all_classes:
        values = [CC[c1][c2] for c2 in all_classes]
        total = sum(values) / 100
        row = ["%8.2f" % (v / total) for v in values]
        print "{:8}| {}".format(c1, ' '.join(row))


def build_dictionary(abstracts):
    '''
    Returns a mapping of all the words in dataset to an index
    '''
    all_words = set()
    for abstract in abstracts:
        words = abstract.split(' ')
        all_words.update(words)
    return dict((v, i) for i, v in enumerate(all_words))


def extract_features(abstracts):
    '''
    Converts an abstract paragraph to a list of features (words)
    '''
    return map(str.split, abstracts)


def compress_classes(categories):
    '''
    Converts classes to indices of a class array
    '''
    classes = list(set(categories))
    mapping = dict((c, i) for i, c in enumerate(classes))
    output = [mapping[c] for c in categories]
    return output, classes


def load_train_data():
    '''
    Loads the set of train data and results
    '''

    with open('train_input.csv') as fp:
        reader = csv.reader(fp)
        train_input = list(reader)[1:]

    with open('train_output.csv') as fp:
        reader = csv.reader(fp)
        train_output = list(reader)[1:]

    abstracts = map(lambda x: x[1], train_input)
    categories = map(lambda x: x[1], train_output)

    bad_indices = set(i for i, c in enumerate(categories) if c == "category")
    abstracts = [a for i, a in enumerate(abstracts) if i not in bad_indices]
    categories = [c for i, c in enumerate(categories) if i not in bad_indices]

    return abstracts, categories


def load_test_data():
    '''
    Loads the set of test data
    '''
    # Load test data
    with open('test_input.csv') as fp:
        reader = csv.reader(fp)
        train_input = list(reader)[1:]

    abstracts = map(lambda x: x[1], train_input)

    return abstracts


def write_test_output(output_data):
    '''
    Writes a set of predictions to file
    '''
    with open('test_output.csv', 'wb') as fp:
        writer = csv.writer(fp, quoting=csv.QUOTE_ALL)
        writer.writerow(['id', 'category'])  # write header
        for i, category in enumerate(output_data):
            writer.writerow((str(i), category))
