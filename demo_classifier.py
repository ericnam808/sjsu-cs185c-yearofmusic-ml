"""
Script for demonstrating classifiers.

:author: M. Layman
"""

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfTransformer

import heapq, itertools, threading

def _sift_results_(i, j, n=10, threshold=0.00001):
    """
    Filters out results less than the threshold.

    i: Iterator of indexes.
    j: Iterator of probabilities.
    n: Max number of results listed. Must satisfy n > 0 (default = 10).
    threshold: Minimum threshold items in 'j' must meet to be returned. 
    
    :return: Returns a descending list of tuples (i, j-th value) of the 
    the top-n j-values.
    """
    assert(len(i) == len(j))
    assert(n >= 0)
    
    _data = itertools.ifilter(
        lambda v: v[1] >= threshold,
        itertools.imap(
            lambda a,b: (a,b), i, j))

    return heapq.nlargest(n, _data, key=lambda v: v[1])
    
def build_classifier(file_path, threshold):
    
    print "[INFO] Loading classifier '%s'..." % file_path,
    
    clf_obj = joblib.load(file_path)
    
    clf = clf_obj['classifier']
    count_vect = clf_obj['vectorizer']
    cls = clf_obj['class_list']
    tfidf_transformer = TfidfTransformer()
    
    print "Done!"

    def _w_(str_):
        
        # original vect is a CountVectorizer
        interm_s = count_vect.transform([str_]).toarray()
        
        # Vectorize by tf-idf
        s = tfidf_transformer.fit_transform(interm_s)
    
        print "Searching for '%s'..." % str_
        r = clf.predict_proba(s)
        x = _sift_results_(cls, r[0], n=10, threshold=threshold)

        if (len(x) == 0):
            return {
                'search_str': str_,
                'result': ["No results found."] }

        y = []
        for l in itertools.imap(lambda v: v, x):
            y.append((l[0], "%.3f%%" % (100.0 * l[1])))
            
        return {
            'search_str': str_,
            'result': y }

    return _w_


class Worker(threading.Thread):

    def __init__(self, clf, search):
        super(Worker, self).__init__()
        self.clf = clf
        self.search = search
        # CRAP: design
        self.result = []

    def run(self):
        self.result.append(
            self.clf(self.search))
        
def compose_searcher(clf_fn):
    
    clf = build_classifier(clf_fn, threshold=0.01)
    
    def _w_(text):

        clf_thrd = Worker(clf, text)
        clf_thrd.start()
        clf_thrd.join()

        # display results
        for obj in clf_thrd.result:
            for rslt in obj['result']:
                print rslt
            
    return _w_

search = compose_searcher(
    r'classifiers/MultinomialNB-Artist_Title Year Classifier.pkl')

print "\nIn the command line, type search('text'), replacing -text- with your search string."
print "Search is performed by word (space-delimited). Only the top 10 results are listed."

