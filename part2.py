from __future__ import division

from preprocessing import *
from classifiers import *
from utils import *

import cPickle as pickle
import random
random.seed(0)

#-------------
# Prepare data
# ------------
abstracts, categories = load_train_data()
# clean_abstracts = preprocess(abstracts, train=True)

# # IG pruning
# examples = extract_features(clean_abstracts)
# IG = compute_IG(examples, categories)
# IG_words = sorted(IG, key=lambda x: IG[x])
# threshold = int(len(IG_words) * 0.06)
# word_remover = word_remove_factory(IG_words[:threshold])
# clean_abstracts = map(word_remover, clean_abstracts)

with open('clean_abstracts.pickle') as fp:
    clean_abstracts = pickle.load(fp)

# # Shuffle the data
# data = zip(clean_abstracts, categories)
# random.shuffle(data)
# clean_abstracts[:], categories[:] = zip(*data)

examples = extract_features(clean_abstracts)

#------------
# K-fold test
# -----------

predictions = []
valid_success = []
train_success = []
for data in CrossValidation(examples, categories, k=10):
    train_data, train_result, test_data, test_result = data

    classifier = AdaBoost_SAMME()
    classifier.fit(train_data, train_result)

    guesses = map(classifier.predict, test_data)
    correct = filter(lambda x: x[0] == x[1], zip(guesses, test_result))
    ratio = len(correct) / len(test_result)
    print "Validation success rate:", ratio
    valid_success.append(ratio)
    predictions.extend(guesses)

    guesses = map(classifier.predict, train_data)
    correct = filter(lambda x: x[0] == x[1], zip(guesses, train_result))
    ratio = len(correct) / len(train_result)
    train_success.append(ratio)
    print "Training success rate:", ratio

train_ratio = sum(train_success) / len(train_success) * 100

valid_ratio = sum(valid_success) / len(valid_success) * 100



#-----------
# Test cases
# ----------

# test_data = preprocess(load_test_data())
# test_examples = extract_features(test_data)

# classifier = AdaBoost_SAMME()
# classifier.fit(examples, categories)
# guesses = map(classifier.predict, test_examples)

# write_test_output(guesses)
