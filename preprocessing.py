from collections import Counter
from porter_stemmer import *
from utils import *
import re


# list of stop words (inspired by most-used words in dataset)
CITE_RE = re.compile(r'\\cite{.*?}')
URL_RE = re.compile("(?:ftp|https?):\/\/[ \da-z\.-]+\.[a-z\.]{2,6}(?:[\/\w\.-]*)*\/?")
EQUATION_RE = re.compile(r'(?<!\\)(?:((?<!\$)\${1,2}(?!\$))|(\\\()|(\\\[)|(\\begin\{equation\}))(.*?(\g<1>)?.*?)(?<!\\)(?(1)(?<!\$)\1(?!\$)|(?(2)\\\)|(?(3)\\\]|\\end\{equation\})))')
STYLE1_RE = re.compile(r'\\\w+{(.*?)}')
STYLE2_RE = re.compile(r'{\\\w+ (.*?)}')
SPECIALCHAR_RE = re.compile(r'\\.')
BRACKET_RE = re.compile('[\[\]\(\){}<>]')
PUNCTUATION_RE = re.compile("[.,?!]")
NUMBER_RE = re.compile('\d+')

SEPARATOR_RE = re.compile('[ -/]')

STOP_WORDS = set([
    'the', 'is', 'at', 'which', 'on', 'of', 'and', 'to', 'we', 'us', 'for',
    'that', 'this', 'with', 'are', 'by', 'as', 'an', 'be', 'from', 'can',
    'it', 'its', 'our', 'these', 'or', 'have', 'has', 'their', 'in', 'a'
])


def word_remove_factory(to_remove):
    to_remove = set(to_remove)
    def word_remover(abstract):
        words = [w for w in abstract.split(' ') if w not in to_remove]
        return ' '.join(words)
    return word_remover


def build_word_counts(abstracts):
    words = []
    for abstract in abstracts:
        words.extend(abstract.split(' '))
    return Counter(words)


def preprocess(abstracts, train=False):
    clean_abstracts = map(clean_up, abstracts)

    if train:
        # Remove words that appear only once
        # Reduces dictinary size from 85k to 47k
        word_count = build_word_counts(clean_abstracts)
        rare_words = set(w for w, c in word_count.items() if c <= 2)
        word_remover = word_remove_factory(rare_words)
        clean_abstracts = map(word_remover, clean_abstracts)

    return clean_abstracts


def clean_up(abstract):
    '''
    Takes an abstract as input and cleans them up.
    Brings the dictionary size from 428,207 words to 85,012 (20%)
    '''
    # remove caps
    abstract = abstract.lower()

    # replace http://... with LINK
    abstract = URL_RE.sub('LINK', abstract)

    # replace \cite{...} with CITE (for citation)
    abstract = CITE_RE.sub('CITE', abstract)

    # replace latex formula ($...$ and others) with FORMULA
    abstract = EQUATION_RE.sub('FORMULA', abstract)

    # Remove punctuation
    abstract = PUNCTUATION_RE.sub('', abstract)

    # Removing styling commands e.g \emph{text}
    abstract = STYLE1_RE.sub(r'\1', abstract)
    abstract = STYLE2_RE.sub(r'\1', abstract)

    # # # Remove special character commands e.g. Turing\'s
    # abstract = SPECIALCHAR_RE.sub('', abstract)

    # Remove all extra bracketing e.g. () [] {} <>
    abstract = BRACKET_RE.sub('', abstract)

    # # Replace numbers with NUMBER
    # abstract = NUMBER_RE.sub('NUMBER', abstract)

    # Remove stop words and stem the words, then put it back together
    words = SEPARATOR_RE.split(abstract)
    words = filter(lambda x: len(x) > 1, words)
    words = filter(lambda x: x not in STOP_WORDS, words)
    # words = map(porter_stem, words)
    abstract = ' '.join(words)

    return abstract
