from __future__ import division

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import svm
from preprocessing import *
import cPickle as pickle
from utils import *

#-------------
# Prepare data
# ------------
abstracts, categories = load_train_data()
clean_abstracts = preprocess(abstracts, train=True)

# IG pruning
examples = extract_features(clean_abstracts)
IG = compute_IG(examples, categories)
IG_words = sorted(IG, key=lambda x: IG[x])
threshold = int(len(IG_words) * 0.05)
word_remover = word_remove_factory(IG_words[:threshold])
clean_abstracts = map(word_remover, clean_abstracts)

# with open('clean_abstracts.pickle') as fp:
#     clean_abstracts = pickle.load(fp)

#------------
# K-fold test
# -----------

success_rates = []
for data in CrossValidation(clean_abstracts, categories, k=5):
    train_data, train_result, test_data, test_result = data

    clf = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        ('clf', SGDClassifier(alpha=7.7e-6, n_iter=13)),
    ])

    clf.fit(train_data, train_result)
    guesses = clf.predict(test_data)

    correct = filter(lambda x: x[0] == x[1], zip(guesses, test_result))
    ratio = len(correct) / len(test_result)
    success_rates.append(ratio)

success_ratio = sum(success_rates) / len(success_rates) * 100
print "Average success rate:", success_ratio


# -----------
# Test cases
# -----------

# test_data = preprocess(load_test_data())

# clf = Pipeline([
#         ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
#         ('clf', SGDClassifier(alpha=7.7e-6, n_iter=13)),
# ])

# clf.fit(clean_abstracts, categories)
# guesses = clf.predict(test_data)

# write_test_output(guesses)
