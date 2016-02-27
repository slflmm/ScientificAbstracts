The code base is entirely written in Python 2.7, and is seperated in multiple files.
Most of the code was written to be runnable on PyPy for a performence boost.
Almost all pieces (preprocessing, crossvalidation, etc), run in under 5m using PyPy.
The only file that requires CPython is part3.py which uses SKLearn.


-- utils.py
This file contains various utility functions used throughout.
Things like a CrossValidation utility, and functions to load/store datasets.
Each function has a docstring giving a short description of what it does.


-- preprocessing.py
This file preprocesses the abstracts, cleaning up each abstract.
It does things such as removing LaTeX markers, URLs and punctuation.
It also lowercases everything and stems the words. At the end, it
simply outputs abstracts again.


-- porter_stemmer.py
A stemming algorithm we have implemented based on the classic paper
by M.F. Porter from the 80's which still performs really well.
This is used inside the preprocessing script.


-- classifiers.py
This file contains the implement classifiers, in an OO model.
We have implemented NaiveBayes and a modified AdaBoost classifier
for multi-class problems. There's also a weak learner used for boosting.
Each classifier has a fit and predict method.


-- part1.py / part2.py / part3.py
These files are example run files for the 3 parts of the project.
part1 shows NaiveBayes, part2 shows Boosting and part3 shows SVM.
Each file loads the data, preprocesses it, extracts featuers, then
either runs crossvalidation or test set classification.


-- transforms.py
This is a file that implements a couple transforms such as tf-idf,
which were never actually used in the first two parts of the project.