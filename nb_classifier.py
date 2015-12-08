"""
M.Layman
E Nam
CS 185 HW 5 - FA 2015

Naive bayesian classifier using sklearn.
"""

from numpy import *
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
#from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics

import csv

def splice_csv_columns(data_filepath, columns):

    features = []
    with open(data_filepath, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:   
            features.append(
                map(lambda i: line[i], columns))
                
    return features

def splice_columns(arr, indexes):
    if len(indexes) == 1:
        return map(lambda ar: ar[indexes[0]].lower(), arr)
    return map(lambda ar: reduce(lambda x,y: x + ' ' + y, map(lambda i: ar[i].lower(), indexes)), arr)


def build_unique_list(iter_):
    return list(set(iter_))

def index_list(iter_, codes):
    return map(lambda v: codes.index(v), iter_)

def train_classifer(classifer, training_file, test_file, columns, class_col, data_cols):
    
    raw = splice_csv_columns(
        training_file, columns)

    raw_classes = splice_columns(raw,[class_col])
    training_features = splice_columns(raw, [1,2])
    
    print raw_classes[:10]
    print training_features[:10]
    
    count_vect = CountVectorizer()
    # Vectorize by tf-idf
    tfidf_transformer = TfidfTransformer()
    X_train_counts = count_vect.fit_transform(training_features)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    classifer.fit(X_train_tfidf.toarray(), raw_classes)

    test_raw = splice_csv_columns(
        test_file, columns)

    test_classes = splice_columns(test_raw,[class_col])
    test_features = splice_columns(test_raw, [1,2])

    X_new_counts = count_vect.transform(test_features)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    
    class_names = build_unique_list(raw_classes)
    #class_vect = index_list(raw_classes, class_names)

    results = classifer.predict(X_new_tfidf.toarray())
    
    report = metrics.classification_report(test_classes, results, labels=None, target_names=class_names)
    
    print report
    return report

def main():

    columns = [0,2,3]
    class_col = 0
    feature_cols = [2,3]

    def execute_test(title, clf, output_filepath):

        print '## Executing Classifer "%s" ##' % title

        print 'Training...'
        results = train_classifer(
            clf,
            r'data/training_tracks_MIN5.csv',
            r'data/test_tracks_MIN5.csv',
            columns,
            class_col,
            feature_cols)

        print 'Prediction...'

        print 'Results...'
        with open(output_filepath, 'w') as outfile: 
            outfile.write(results)
                
    classifers = [
        ('GaussianNB - Gaussian Naive Bayes',
#             SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42),
                GaussianNB(),
                r'results/nb_result_2.txt') ]
    
    # execute test
    map(lambda x: execute_test(x[0],x[1],x[2]), classifers)
    
main()
