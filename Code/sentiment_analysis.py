from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import argparse
from scipy.sparse import hstack
from sklearn.ensemble import ExtraTreesRegressor
import operator
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from itertools import izip

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())
    def fit(self, X):
        return self
    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

def get_parser():
    parser = argparse.ArgumentParser(description="Sentiment Polarity and Intensity Regression")
    parser.add_argument('-train', type=str, required=True, help='train file')
    parser.add_argument('-test', type=str, required=True, nargs='+',help='test file')
    parser.add_argument('-w2vec', type=str, required=True, nargs='+',help='w2vec file')
    return parser

def loadl(f):
    a = []
    with open(f,"r") as fid:
        for l in fid:
            a.append(l.lower().strip('\n'))
    return a

def loadmpqa(f):
    pos = []
    neg = []
    neu = []
    with open(f,"r") as fid:
        for l in fid:
            w = l.lower().strip('\n').split('\t')
            if w[1] == '1':
                pos.append(w[0])
            elif w[1] == '-1':
                neg.append(w[0])
            elif w[1] == '0':
                neu.append(w[0])

    return pos, neg, neu

def loadmpqas(f):
    pos = []
    neu = []
    with open(f,"r") as fid:
        for l in fid:
            w = l.lower().strip('\n').split('\t')
            if w[1] == '1':
                pos.append(w[0])
            elif w[1] == '0':
                neu.append(w[0])
    return pos, neu

def getLexicon(msgs_train, msgs_test):
    print 'getLexicon'
    fol = 'lexicals/'
    a = loadl(fol + 'loughran_constraining.tsv')
    b = loadl(fol + 'loughran_letigious.tsv')
    c = loadl(fol + 'loughran_negative.tsv')
    d = loadl(fol + 'loughran_positive.tsv')
    e = loadl(fol + 'loughran_uncertainty.tsv')
    t = loadl(fol + 'loughran_modal.tsv')
    q = loadl(fol + 'loughran_harvard.tsv')
    p_pos, p_neg, p_neu = loadmpqa('mpqa_polarity')
    s_pos, s_neu = loadmpqas('mpqa_subjectivity')
    tlist = []
    for msg in msgs_train:
        m = set(msg.split())
        a_s = set(a).intersection(m)
        b_s = set(b).intersection(m)
        c_s = set(c).intersection(m)
        d_s = set(d).intersection(m)
        e_s = set(e).intersection(m)
        f_s = set(p_pos).intersection(m)
        g_s = set(p_neg).intersection(m)
        h_s = set(p_neu).intersection(m)
        l_s = set(s_pos).intersection(m)
        m_s = set(s_neu).intersection(m)
        t_s = set(t).intersection(m)
        q_s = set(q).intersection(m)
        msg_score = 0.0
        lg_score = 0.0
        if len(f_s) > 0 or len(g_s) > 0:
            msg_score = float(len(f_s)) - float(len(g_s))/float(len(m))
        if len(c_s) > 0 or len(d_s) > 0:
            lg_score = float(len(d_s)) - float(len(c_s) ) /float(len(m))
        dic = {'lg_score':lg_score, 'laug_a': 1 if a_s else 0 , 'laug_b': 1 if b_s else 0, 'laug_c': 1 if c_s else 0, 'laug_d':1 if d_s else 0, 'laug_e':1 if e_s else 0}#, 
        tlist.append(dic)
    flist = []
    for msg in msgs_test:
        m = set(msg.split())
        a_s = set(a).intersection(m)
        b_s = set(b).intersection(m)
        c_s = set(c).intersection(m)
        d_s = set(d).intersection(m)
        e_s = set(e).intersection(m)
        f_s = set(p_pos).intersection(m)
        g_s = set(p_neg).intersection(m)
        h_s = set(p_neu).intersection(m)
        l_s = set(s_pos).intersection(m)
        m_s = set(s_neu).intersection(m)
        t_s = set(t).intersection(m)
        q_s = set(q).intersection(m)
        msg_score = 0.0
        lg_score = 0.0
        if len(f_s) > 0 or len(g_s) > 0:
            msg_score = float(len(f_s)) - float(len(g_s))/float(len(m))
        if len(c_s) > 0 or len(d_s) > 0:
            lg_score = float(len(d_s)) - float(len(c_s) ) /float(len(m))
        flist.append(dic)
    vec = DictVectorizer(sparse=False)
    vec.fit(tlist)
    X_train_lex = vec.transform(tlist)
    X_test_lex = vec.transform(flist)
    return X_train_lex, X_test_lex


def  train_predict(labels_train, msgs_train, labels_test, msgs_test, cashtags_train, cashtags_test, embeddings):
    print "loading features..."
    Y_train = np.array([float(x) for x in labels_train])
    Y_test = np.array([float(x) for x in labels_test])
    X_train_bow = []
    X_train_boe = []
    X_test_bow = []
    X_test_boe = []
    X_train = []
    X_test = []
    X_train_bos = []
    X_test_bos = []
    print 'train: ', len(Y_train), 'test:', len(Y_test)
    vec = CountVectorizer(ngram_range=(1,1))
    vec.fit(msgs_train)
    X_train_bow = vec.transform(msgs_train).toarray()
    X_test_bow = vec.transform(msgs_test).toarray()
    model = gensim.models.Word2Vec.load(embeddings) 
    w2v = dict(zip(model.index2word, model.syn0)) 
    vec = MeanEmbeddingVectorizer(w2v)
    vec.fit(msgs_train)
    X_train_boe = vec.transform(msgs_train)#.toarray()
    X_test_boe = vec.transform(msgs_test)#.toarray()
    print 'len x_Train_bow, X_test_bow', X_train_bow.shape , X_test_bow.shape
    print 'len x_Train_boe, X_test_boe', X_train_boe.shape , X_test_boe.shape
    X_train_lex, X_test_lex = getLexicon(msgs_train, msgs_test)
    X_train = np.concatenate((X_train_bow,X_train_boe, X_train_lex), axis=1)
    X_test = np.concatenate((X_test_bow,X_test_boe, X_test_lex), axis=1)
    print 'len x_Train_, X_test_', X_train.shape , X_test.shape
    clf = ExtraTreesRegressor(n_estimators=200, n_jobs=-1)
    #clf = MLPRegressor(hidden_layer_sizes=(5,) ,verbose=True)
    #clf = SVR(kernel='linear',C=1.0, epsilon=0.2)
    print "fiting.. extra trees."
    print Y_train
    clf.fit(X_train, Y_train)
    print "testing..."
    y_hat = clf.predict(X_test)
    res_dict = {}
    mae = mean_absolute_error(Y_test, y_hat)
    cos = cosine_similarity(Y_test, y_hat)
    print 'MAE: ', mae, '\tCOSINE:', cos
    return mae, cos


def readfile(path):
    labels = []
    msgs = []
    cashtags = []
    with open(path,"r") as fid:
        for l in fid:
            splt = l.strip('\n').lower().split("\t")
            labels.append(splt[3])
            msgs.append(splt[2])
            cashtags.append(splt[0].split('_')[1].replace('$',''))
    return labels, msgs, cashtags


def main():
    parser = get_parser()
    args = parser.parse_args()    
    train_file = args.train
    test_file = args.test[0]
    embeddings = args.w2vec[0]
    labels_train, msgs_train, cashtags_train = readfile(train_file)
    labels_test, msgs_test, cashtags_test = readfile(test_file)
    train_predict(labels_train, msgs_train, labels_test, msgs_test, cashtags_train, cashtags_test, embeddings)


if __name__ == "__main__":
    main()
