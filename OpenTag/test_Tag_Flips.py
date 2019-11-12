import pytest
import numpy as np
import Tag_Flips

learner_a = [[np.array([0., 0., 1., 0.]), np.array([0., 0., 1., 0.]), np.array([0., 0., 1., 0.])],\
            [np.array([0., 0., 1., 0.]), np.array([0., 0., 1., 0.]), np.array([0., 0., 1., 0.])]]    
learner_b = [[np.array([0., 0., 1., 0.]), np.array([0., 0., 1., 0.]), np.array([0., 0., 1., 0.])],\
            [np.array([0., 1., 0., 0.]), np.array([0., 0., 1., 0.]), np.array([1., 0., 0., 0.])]]
learner_c = [[np.array([0., 0., 1., 0.]), np.array([0., 0., 1., 0.]), np.array([0., 0., 1., 0.])],\
            [np.array([0., 1., 0., 0.]), np.array([1., 0., 0., 0.]), np.array([1., 0., 0., 0.])]]
learners = [learner_a, learner_b, learner_c]

tf = Tag_Flips.Tag_Flips()
tf.insert_predictions(learners)

def test_arrays_diff():
    a = np.array([0., 0., 1., 0.])
    b = np.array([0., 1., 1., 0.])
    assert tf.arrays_diff(a, b) == 1

def test_sentence_difference():
    # comparison between the tags of two sentences
    a = [np.array([0., 0., 1., 0.]), np.array([1., 1., 1., 0.]), np.array([0., 0., 1., 0.])]
    b = [np.array([0., 1., 1., 0.]), np.array([0., 0., 1., 0.]), np.array([0., 0., 1., 1.])]
    assert tf.sentence_difference(a, b) == 3

    a = [np.array([0., 1., 1., 0.]), np.array([0., 0., 1., 0.]), np.array([0., 0., 1., 0.])]
    b = [np.array([0., 1., 1., 0.]), np.array([0., 0., 1., 0.]), np.array([0., 0., 1., 0.])]
    assert tf.sentence_difference(a, b) == 0

def test_more_learners_difference_on_sentence():
    # each learner tags two sentences: each row is one sentence
    # first one is the tagged the same, the second is different in three points
    assert tf.more_learners_difference_on_sentence(0) == 0
    assert tf.more_learners_difference_on_sentence(1) == 3

def test_more_learners_difference_on_all_sentences():
    # each learner tags two sentences: each row is one sentence
    # first one is the tagged the same, the second is different in three points
    assert tf.more_learners_difference_on_all_sentences() == {0: 0, 1: 3}

def test_get_new_train_set_indices():
    assert tf.get_new_train_set_indices(1) == [1]
