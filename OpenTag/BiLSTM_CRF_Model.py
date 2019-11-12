import numpy as np
from keras.models import Model, Input, load_model
from keras_contrib.layers import CRF
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib import losses
from keras_contrib.utils import save_load_utils
from keras_contrib.metrics import crf_accuracy
import random
import h5py

class BiLSTM_CRF_Model:
    
    def __init__(self, train_test_obj, optimizer="rmsprop", loss=losses.crf_loss, drop=0.4, hidden_layer_dim=20, verbose = True):
        self.train_test_obj = train_test_obj
        self.optimizer = optimizer
        self.loss = loss
        self.drop = drop
        self.hidden_layer_dim = hidden_layer_dim
        self.verbose = verbose

        self.build_model(train_test_obj.get_max_len(), train_test_obj.get_number_words(), train_test_obj.get_number_tags())
        
    def build_model(self, sent_max_len, num_words, num_tags):
        if self.verbose: print("[INFO] Setting up model\n")
        inputs = Input(shape = (sent_max_len,))
        model = Embedding(input_dim = num_words + 1, output_dim = self.hidden_layer_dim, input_length = sent_max_len, mask_zero=True)(inputs)
        model = Bidirectional(LSTM(units = sent_max_len, return_sequences=True, recurrent_dropout=self.drop))(model) 
        crf = CRF(num_tags)
        out = crf(model) 
        
        self.model = Model(inputs, out)
        self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = [crf_accuracy])
   
    def train_model(self, epochs, *args, r_stat = 123, validation_split = 0):
        self.set_train_test(*args)
    
        if self.verbose: print("[INFO] Training model\n")
        self.history = self.model.fit(self.X_train, np.array(self.y_train), batch_size=32, epochs=epochs, validation_split = validation_split, verbose = 2)

    def set_train_test(self, *args):
        if len(args)!=1 and len(args)!=2 and len(args)!=4 :
            raise TypeError('set_train_test() takes either 1, 2 or 4 arguments ({} given)' .format(len(args)))

        if len(args) == 1:
            self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_obj.train_test(test_size = args[0])
        elif len(args) == 2:
            self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_obj.train_test(test_size = args[0], r_stat=args[1])
        elif len(args) == 4:
            self.X_train, self.y_train, self.X_test, self.y_test = args

    def predict(self, *args):
        if len(args)!=0 and len(args)!=1:
            raise TypeError('predict() takes either 0 or 1 arguments ({} given)' .format(len(args)))

        if self.verbose: print("[INFO] Predicting:\n")
        if len(args) == 1:
            self.pred = self.model.predict(args[0], verbose=1)
        else:
            self.pred = self.model.predict(self.X_test, verbose=1)
        
    def get_predictions(self):
        return self.pred

    def get_prediction_tags(self):
        out = []
        for pred_i in self.pred:
            out_i = []
            for p in pred_i:
                out_i.append(self.train_test_obj.idx2tag[np.argmax(p)].replace("PAD", "'O'"))
            out.append(out_i)
        return out

    def pretty_print_prediction(self, *args):
        if len(args)!=0 and len(args)!=1:
            raise TypeError('pretty_print_prediction() takes either 0 or 1 arguments ({} given)' .format(len(args)))

        if len(args) == 1 and isinstance(args[0], int):
            idx = args[0]
        elif len(args) == 0:
            # in this case it performs prediction on a random sentence
            idx = random.randint(0,self.y_test.shape[0]-1)
            
        p = np.argmax(self.pred[idx], axis=-1)
        true = np.argmax(self.y_test[idx], -1)
        print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
        for w, t, predd in zip(self.X_test[idx], true, p):
            if self.train_test_obj.words[w-1] != "ENDPAD":
                print("{:15}: {:5} {}".format(self.train_test_obj.words[w-1], self.train_test_obj.tags[t], self.train_test_obj.tags[predd]))

    def save_trained_model(self, name):
        if self.verbose: print("\n[INFO] Saving trained model to '" + name + "'\n")
        save_load_utils.save_all_weights(self.model, name)

    def load_trained_model(self, name):
        if self.verbose: 
            print("\n[INFO] Loading trained model from '" + name +"'")
            print("Performing dummy training in order to be able to load weights:\n")
        self.model.fit(self.X_train[0:5], np.array(self.y_train[0:5]), epochs=1)

        save_load_utils.load_all_weights(self.model, name)
        self.model.compile(optimizer = self.optimizer, loss = self.loss, metrics = [crf_accuracy])
        if self.verbose: print("[INFO] Model is loaded.\n")

    def get_X_train(self):
        return self.X_train
    
    def get_y_train(self):
        return self.y_train

    def get_X_test(self):
        return self.X_test
    
    def get_y_test(self):
        return self.y_test

    def print_summary(self):
        self.model.summary()

    def get_history(self):
        return self.history