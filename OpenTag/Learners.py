from sklearn.model_selection import train_test_split
import numpy as np
import BiLSTM_CRF_Model
import pickle

class Learners:

    def __init__(self, train_test_obj, learners_num):
        self.train_test_obj = train_test_obj
        self.learners_num = learners_num
        self.drop = 1/self.learners_num
        self.define_learners()
        
    def define_learners(self):
        self.learners = [BiLSTM_CRF_Model.BiLSTM_CRF_Model(self.train_test_obj, drop = self.drop*(i+1), verbose=False) for i in range(0,self.learners_num)]

    def set_train_test(self, *args):
        if len(args) not in [1, 2, 4] :
            raise TypeError('set_train_test() takes either 0, 1, 2 or 4 arguments ({} given)' .format(len(args)))

        if len(args) == 1:
            self.X_tr, self.X_te, self.y_tr, self.y_te = self.train_test_obj.train_test(test_size = args[0])
        elif len(args) == 2:
            self.X_tr, self.X_te, self.y_tr, self.y_te = self.train_test_obj.train_test(test_size = args[0], r_stat = args[1])
        elif len(args) == 4:
            self.X_tr, self.y_tr, self.X_te, self.y_te = args

    def train_predictions(self, *args):
        if len(args) != 0:
            self.set_train_test(*args)

        self.predictions = []
        for epochs, learner in enumerate(self.learners):
            print("\n==> LEARNER ", epochs+1, ":")
            learner.train_model(epochs+1, self.X_tr, self.y_tr, self.X_te, self.y_te)
            print("==> PREDICTING:")
            learner.predict()
            self.predictions.append(learner.get_predictions())
        
    def predict(self, x_test_tmp):
        pred_tmp = []
        for learner in self.learners:
            pred_tmp.append(learner.predict(x_test_tmp))
        return pred_tmp

    def pretty_print_prediction(self, learner_idx, *args):
        if learner_idx > self.learners_num:
            raise TypeError("Learner index ({}) bigger than total number: {}!".format(learner_idx, self.learners_num))
        self.learners[learner_idx-1].pretty_print_prediction(*args)

    def get_predictions(self):
        return self.predictions

    def save_predictions(self, name):
        print("\n[INFO] Saving predictions to '" + name + "'\n")
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(self.predictions, f, protocol = pickle.HIGHEST_PROTOCOL)

    def load_predictions(self, name):
        print("\n[INFO] Loading predictions from '" + name + "'\n")
        with open(name + '.pkl', 'rb') as f:
            self.predictions = pickle.load(f)

    def upload_train_test_sets(self, idxs):
        self.X_tr = np.concatenate((self.X_tr, np.asarray( [ self.X_te[val] for val in idxs ] )), axis=0)
        self.y_tr = np.concatenate((self.y_tr, np.asarray( [ self.y_te[val] for val in idxs ] )), axis=0)
        self.X_te = np.delete(self.X_te, idxs, axis=0)
        self.y_te = np.delete(self.y_te, idxs, axis=0)

    def get_test_values_by_index(self, idxs):
        uncertain_X_test = np.asarray( [ self.X_te[val] for val in idxs ] )
        uncertain_y_test = np.asarray( [ self.y_te[val] for val in idxs ] )
        return(uncertain_X_test, uncertain_y_test)

    def get_X_train(self):
        return self.X_tr
    
    def get_y_train(self):
        return self.y_tr

    def get_X_test(self):
        return self.X_te
    
    def get_y_test(self):
        return self.y_te

    def print_learners_info(self):
        print("Learners number:", self.learners_num, "\n")
        [learner.print_model_info() for learner in self.learners]