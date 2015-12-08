"""
M.Layman
E Nam
CS 185 HW 5 - FA 2015

Naive bayesian classifier using sklearn.
"""

from numpy import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import metrics

import csv
import logging as log

log.basicConfig(format='[%(levelname)s] %(message)s', level=log.DEBUG)

def splice_delimited_columns(data_filepath, columns):

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
    return map(
        lambda ar: reduce(
            lambda x,y: x + ' - ' + y,
            map(lambda i: ar[i].lower(), indexes)), arr)

def build_unique_list(iter_):
    return list(set(iter_))

def index_list(iter_, codes):
    return map(lambda v: codes.index(v), iter_)

def train_classifer(classifer, training_file, test_file, columns, class_col, data_cols):

    log.info("Reading training file '%s'..." % training_file)
    raw = splice_delimited_columns(
        training_file, columns)
    log.info("Success! %d raw lines." % len(raw))
    
    log.info('Extracting classes and features...')
    raw_classes = splice_columns(raw,[class_col])
    training_features = splice_columns(raw, data_cols)
    
    print "Raw class samples:" 
    print raw_classes[:3]
    print "Raw training feature samples:"
    print training_features[:3]
    
    # account for weird, non-unicode chars 
    count_vect = CountVectorizer(
        decode_error='replace',
        strip_accents='unicode',
        binary=True)

    X_train_counts = count_vect.fit_transform(training_features)
    log.info('Vocabulary list: %d unique words.' % len(count_vect.get_feature_names()))
    
    # Vectorize by tf-idf
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    log.info('Fitting classifer. This may take some time...')
    classifer.fit(X_train_tfidf.toarray(), raw_classes)
    log.info('Fit successful!')
    
    ### Validation Section ###
    
    log.info('### Start Test Validation ###')
    
    log.info("Reading test file '%s'..." % test_file)
    test_raw = splice_delimited_columns(
        test_file, columns)

    test_classes = splice_columns(test_raw,[class_col])
    test_features = splice_columns(test_raw, data_cols)

    print "Test class samples:" 
    print test_classes[:3]
    print "Test feature samples:"
    print test_features[:3]
    
    log.info('Vectorizing test features...')
    X_new_counts = count_vect.transform(test_features)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    
    log.info('Running predictions. This may also take up your time...')
    results = classifer.predict(X_new_tfidf.toarray())
    log.info('Success! Generating report.')
    
    report = metrics.classification_report(
        test_classes, results, labels=None)
    
    return report

def main():

    columns = [0,2,3]
    class_col = 0
    feature_cols = [1,2]

    def execute_test(title, clf, output_filepath):

        print '## Executing Classifer "%s" ##' % title

        report = train_classifer(
            clf,
            r'data/debug_combo2.txt',
            r'data/debug_combo2_test.txt',
            columns,
            class_col,
            feature_cols)

        print 'Results...'
        print report 

        with open(output_filepath, 'w') as outfile: 
            outfile.write(report)
    
    classifers = [
        ('MultinomialNB - Artist_Title Year Classifier',
                MultinomialNB(alpha=.01),
                r'results/nb_result_2.txt') ]
    
    # execute test
    map(lambda x: execute_test(x[0],x[1],x[2]), classifers)
    
    log.info('Normal program exit.')
main()

