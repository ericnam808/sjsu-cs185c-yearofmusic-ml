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

    raw = splice_delimited_columns(
        training_file, columns)

    raw_classes = splice_columns(raw,[class_col])
    training_features = splice_columns(raw, data_cols)

    print raw_classes[:3]
    print training_features[:3]
    
    count_vect = CountVectorizer(
        decode_error='replace',
        strip_accents='unicode',
        binary=False)

    X_train_counts = count_vect.fit_transform(training_features)

    # Vectorize by tf-idf
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    classifer.fit(X_train_tfidf.toarray(), raw_classes)

    ### Validation Section ###
    
    test_raw = splice_delimited_columns(
        test_file, columns)

    test_classes = splice_columns(test_raw,[class_col])
    test_features = splice_columns(test_raw, data_cols)

    X_new_counts = count_vect.transform(test_features)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    
    class_names = build_unique_list(raw_classes)

    class_vect = index_list(raw_classes, class_names)

    results = classifer.predict(X_new_tfidf.toarray())
    
    report = metrics.classification_report(
        test_classes, results, labels=None)
    
    return report

def main():

    columns = [0,2,3]
    class_col = 0
    feature_cols = [1,2]

    def execute_test(title, clf, output_filepath):

        print '## Executing Classifer "%s" ##' % title

        print 'Training...'
        results = train_classifer(
            clf,
            r'data/debug_combo2.txt',
            r'data/debug_combo2_test.txt',
            columns,
            class_col,
            feature_cols)

        print 'Prediction...'

        print 'Results...'
        print results 

        with open(output_filepath, 'w') as outfile: 
            outfile.write(results)
                
    classifers = [
        ('MultinomialNB - Artist_Title Year Classifier',
                MultinomialNB(alpha=.01),
                r'results/nb_result_2.txt') ]
    
    # execute test
    map(lambda x: execute_test(x[0],x[1],x[2]), classifers)
    
main()
