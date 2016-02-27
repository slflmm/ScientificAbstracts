from __future__ import division
from scipy.sparse import *
import numpy as np

# ---------------
# Helpful methods
# ---------------


def build_dictionary(abstracts):
    '''
    Returns a mapping of all the words in dataset to an index
    '''
    all_words = set()
    for abstract in abstracts:
        words = abstract.split(' ')
        all_words.update(words)
    return dict((v, i) for i, v in enumerate(all_words))


# ---------------
# Bag of Words
# ---------------

def bag_of_words(abstracts):
    dictionary = build_dictionary(abstracts)
    return create_bow_matrix(abstracts, dictionary)


def create_bow_matrix(abstracts, dictionary):
    '''
    Returns matrix of size len(abstracts) x len(dictionary).
    For each abstract, a vector of the counts of all words in the dictionary.
    '''
    bow_matrix = dok_matrix((len(abstracts), len(dictionary)))
    for i, abstract in enumerate(abstracts):
        for word in abstract.split(' '):
            bow_matrix[i, dictionary[word]] += 1
    return bow_matrix.tocsr()


# ---------------
# TF-IDF features
# ---------------

def tf(abstracts, dictionary):
    '''
    Returns the term frequency (TF).
    (How frequently a term occurs in a document.)
    Normalized by document length.
    '''
    tf = create_bow_matrix(abstracts, dictionary)
    abstract_sizes = [abstract.count(' ') + 1 for abstract in abstracts]
    scale_factor = diags([[1/s for s in abstract_sizes]], [0])
    return scale_factor * tf


def idf(abstracts, dictionary):
    '''
    Returns the inverse document frequency (IDF).
    (How important a term is -- a vector of length len(dictionary))
    log(Total number of documents / Number of documents with term in it).
    '''
    doc_count = np.zeros(len(dictionary))
    for i, word in enumerate(dictionary):
        for abstract in abstracts:
            if word in abstract:
                doc_count[i] += 1
    scale = np.log(len(abstracts) / doc_count)
    return diags([scale], [0])


def tf_idf(abstracts):
    '''
    Returns the TF-IDF features representation of abstracts.
    This should only be used on preprocessed abstracts.
    '''
    dictionary = build_dictionary(abstracts)
    return tf(abstracts, dictionary) * idf(abstracts, dictionary)


# ----------------------------
# Relevance-frequency features
# ----------------------------
