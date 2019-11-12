import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
np.random.seed(1337)

import Train_Test
import BiLSTM_CRF_Model
import Learners
import Tag_Flips
import pickle
import random
from sklearn.model_selection import train_test_split

with open("../saved_things/tagged_sentences_100000.pkl", 'rb') as f:
        tagged_sentences = pickle.load(f)
with open("../saved_things/words_100000.pkl", 'rb') as f:
        words = pickle.load(f)
tags = ['O', 'B-aroma', 'B-taste', 'B-fruit', 'I-aroma', 'I-taste', 'I-fruit']

tr_ts = Train_Test.Train_Test(tagged_sentences, tags, words)
x, xt, y, yt = tr_ts.train_test(test_size = 0.7, r_stat=123)

def pretty_print_sent(*args):
        if len(args) == 2:
                X, y = args
        elif len(args) == 3:
                X, y, idx = args
                y = y[idx]
                X = X[idx]

        true = np.argmax(y, -1)
        print("{:15}||{:5}".format("Word", "True"))
        for w, t in zip(X, true):
            if words[w-1] != "ENDPAD":
                print("{:15}: {:5}".format(words[w-1], tags[t]))

def call_BiLSTM_CRF_single_model(train_test_set, epochs):
        model = BiLSTM_CRF_Model.BiLSTM_CRF_Model(train_test_obj = tr_ts)
        model.set_train_test(0.5)
        # three poxibilities to train_model (or to use set_train_test):
        # 1)
        #model.train_model(epochs, 0.5)
        # 2)
        #model.train_model(epochs, 0.2, 234)
        # 3)
        #model.train_model(epochs, x, y, xt, yt)
        
        #model.save_trained_model("saved_objects/.h5")
        model.load_trained_model("saved_objects/keras_model_new.h5") # model trained with 6 epochs
        model.predict()
        #model.pretty_print_prediction(3)
        model.pretty_print_prediction()

def call_learners(train_test_set, learners_num):
        obj = Learners.Learners(tr_ts, learners_num)
        # three poxibilities for train_predictions (or to use set_train_test):
        # 1)
        obj.train_predictions(0.7)
        # 2)
        #obj.train_predictions(0.3, 123)
        # 3)
        #obj.train_predictions(x, y, xt, yt)

        ###obj.save_predictions(name = "saved_objects/")
        #obj.load_predictions(name = "saved_objects/predictions")
        predictions = obj.get_predictions()

        tf = Tag_Flips.Tag_Flips()
        tf.compute_Q(predictions)
        incert = tf.get_more_uncertain_sentences(sent_num =  5)
        print(incert)
        ###tf.save_Q(name = "saved_objects/Q_computed")
        #tf.load_Q(name = "saved_objects/Q_computed")
        obj.set_train_test(x, y, xt, yt)
        print(np.array(obj.get_X_train()).shape)
        U = tf.get_new_train_set_indices(threshold = 5)
        print(len(U))
        obj.upload_train_test_sets(idxs = U)
        print(np.array(obj.get_X_train()).shape)
        obj.pretty_print_prediction(learner_idx = 1)

def call_learners_flips(train_test_set, learners_num, cycles):
        obj = Learners.Learners(tr_ts, learners_num)
        obj.set_train_test(0.7)
        tf = Tag_Flips.Tag_Flips()

        for _ in range(0, cycles):
                obj.train_predictions()
                pred = obj.get_predictions()
                tf.compute_Q(learners_predictions = pred)
                incert = tf.get_more_uncertain_sentences(sent_num = 5)
                print(incert)
                obj.pretty_print_prediction(2)
               
                U = tf.get_new_train_set_indices(threshold = 5)
                print("train shape", np.array(obj.get_X_train()).shape)
                print("test shape", np.array(obj.get_X_test()).shape)
                obj.upload_train_test_sets(idxs = U)
                print("train shape", np.array(obj.get_X_train()).shape)
                print("test shape", np.array(obj.get_X_test()).shape)
        
        # I can finally get the more uncertain sentences
        x_uncert, y_uncert = obj.get_test_values_by_index(idxs = U)
        pretty_print_sent(x_uncert, y_uncert)

def call_Q():
        tf = Tag_Flips.Tag_Flips()
        tf.load_Q(name = "saved_objects/Q_computed")
        incert = tf.get_more_uncertain_sentences(sent_num = 5)
        print(incert)
        #worst = next(iter(incert))
        #print(worst)



#pretty_print_sent(x[4], y[4]) # or pretty_print_sent(x, y, 4)
#call_BiLSTM_CRF_single_model(tr_ts, epochs=2)
#call_learners(tr_ts, learners_num = 2)
call_learners_flips(tr_ts, learners_num = 2, cycles = 2)
#call_Q()