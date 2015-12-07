from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

import heapq, itertools

def _sift_results_(index, proba, n=10, min_limit=0.00001):

    assert(len(index) == len(proba))
    
    _data = itertools.ifilter(
        lambda v: v[1] >= min_limit,
        itertools.imap(
            lambda a,b: (a,b),
            index, proba))

    return heapq.nlargest(n, _data, key=lambda v: v[1])
    
def setup_classifier(file_path):
    
    print "[INFO] Loading classifier '%s'..." % file_path,
    clf_obj = joblib.load(file_path)
    print 'Done!'

    clf = clf_obj['classifier']
    cls = clf_obj['class_list']
    dbo = clf_obj['class_year_map']
    
    print "[INFO] Rehydrating classes...",
    vectorizer = CountVectorizer(
        decode_error='replace',
        strip_accents='unicode')
    vectorized_corpus = vectorizer.fit_transform(cls)
    print "Done!"

    def _w_(str_):
        
        s = vectorizer.transform([str_]).toarray()
        t = vectorizer.inverse_transform(s)
        
        print "Searching for '%s'..." % str_,
        r = clf.predict_proba(s)
        x = _sift_results_(cls, r[0], n=10, min_limit=0.05)

        if (len(x) == 0):
            print "No results found.\n"
            return

        print 'Top-%d results found.\n' % len(x)
        
        for l in itertools.imap(lambda v: v, x):
            print l,
            if (l[0] in dbo):
                print dbo[l[0]]
            else:
                print 'No years found.'
    
    return _w_

#search = setup_classifier(r'classifiers/GNB-Artist Classifer.pkl')
search = setup_classifier(r'classifiers/GNB-Title Classifer.pkl')

print "\nIn the command line, type search('string'), replacing -string- with your search string." 
