import pytest
import numpy as np
import Train_Test

tags = ['O', 'B-geo', 'B-gpe']
sent = [('thousands','NNS','O'),('of','IN','O'),('demonstrators','NNS','O'),('have','VBP','O'),('marched','VBN','O'),('through','IN','O'),('london','NNP','B-geo'),('to','TO','O'),('protest','VB','O'),('the','DT','O'),('war','NN','O'),('in','IN','O'),('iraq','NNP','B-geo'),('and','CC','O'),('demand','VB','O'),('the','DT','O'),('withdrawal','NN','O'),('of','IN','O'),('british','JJ','B-gpe'),('troops','NNS','O'),('from','IN','O'),('that','DT','O'),('country','NN','O')]
sent2 = [('police','NNS','O'),('put','VBD','O'),('the','DT','O'),('number','NN','O'),('of','IN','O'),('marchers','NNS','O'),('at','IN','O'),('10','CD','O'),('while','IN','O'),('organizers','NNS','O'),('claimed','VBD','O'),('it','PRP','O'),('was','VBD','O'),('10,000','CD','O')]
tagged_sentences = [sent, sent2]
words = ['police', 'put', 'the', 'number', 'of', 'marchers', 'at', '10', 'while', 'organizers', 'claimed',\
         'it', 'was', '10,000', 'thousands', 'demonstrators', 'have', 'marched', 'through', 'london', 'to',\
         'protest', 'war', 'in', 'iraq', 'and', 'demand', 'withdrawal', 'british', 'troops', 'from',      \
         'that', 'country']

tr_ts = Train_Test.Train_Test(tagged_sentences, tags, words)

def test_number_tags():
    assert(len(tags) == tr_ts.get_number_tags())

def test_number_words():
    assert(len(words) == tr_ts.get_number_words())

def test_max_len():
    assert(23 == tr_ts.get_max_len())

def test_get_x_y():
    x, y = tr_ts.get_x_y()
    x_actual0 = [15, 5, 16, 17, 18, 19, 20, 21, 22, 3, 23, 24, 25, 26, 27, 3, 28, 5, 29, 30, 31, 32, 33]
    y_actual0 = [[1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.],\
                 [0., 1., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.],\
                 [0., 1., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.],\
                 [0., 0., 1.], [1., 0., 0.], [1., 0., 0.], [1., 0., 0.],[1., 0., 0.]]
    assert (y[0] == y_actual0).all()
    assert (x[0] == x_actual0).all()

def test_train_test():
    x, xt, y, yt = tr_ts.train_test(test_size = 0.5, r_stat=123)
    x_train_expected = np.array([15, 5, 16, 17, 18, 19, 20, 21, 22, 3, 23, 24, 25, 26, 27, 3, 28, 5, 29, 30, 31, 32, 33])
    x_test_expected = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    y_train_expected = np.array([[1., 0., 0.], [1., 0., 0.],  [1., 0., 0.],  [1., 0., 0.],
                                        [1., 0., 0.],  [1., 0., 0.],  [0., 1., 0.],  [1., 0., 0.],
                                        [1., 0., 0.],  [1., 0., 0.],  [1., 0., 0.],  [1., 0., 0.],
                                        [0., 1., 0.],  [1., 0., 0.],  [1., 0., 0.],  [1., 0., 0.],
                                        [1., 0., 0.],  [1., 0., 0.],  [0., 0., 1.],  [1., 0., 0.],
                                        [1., 0., 0.],  [1., 0., 0.],  [1., 0., 0.]])
    y_test_expected = np.array([[1., 0., 0.],  [1., 0., 0.],  [1., 0., 0.],
                                [1., 0., 0.],  [1., 0., 0.],  [1., 0., 0.],  [1., 0., 0.],
                                [1., 0., 0.],  [1., 0., 0.],  [1., 0., 0.],  [1., 0., 0.],
                                [1., 0., 0.],  [1., 0., 0.],  [1., 0., 0.],  [1., 0., 0.],
                                [1., 0., 0.],  [1., 0., 0.],  [1., 0., 0.],  [1., 0., 0.],
                                [1., 0., 0.],  [1., 0., 0.],  [1., 0., 0.],  [1., 0., 0.]])
    assert(x == x_train_expected).all()
    assert(xt == x_test_expected).all()
    assert(y == y_train_expected).all()
    assert(yt == y_test_expected).all()
