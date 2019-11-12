import operator
import pickle

class Tag_Flips:
    
    def __init__(self):
        pass

    def insert_predictions(self, learners_predictions):
        self.learners_predictions = learners_predictions
        self.num_sentences = len(self.learners_predictions[0])

    def compute_Q(self, learners_predictions):
        self.insert_predictions(learners_predictions)
        self.more_learners_difference_on_all_sentences()

    def get_Q_value(self):
        return self.Q

    def get_more_uncertain_sentences(self, sent_num):
        return dict(sorted(self.Q.items(), key=operator.itemgetter(1), reverse=True)[:sent_num])

    @staticmethod
    def arrays_diff(arr1, arr2):
        return 1 if (arr1 != arr2).any() else 0

    def sentence_difference(self, list1, list2): # a sentence is list of np arrays
        Q = 0
        for arr1, arr2 in zip(list1, list2):
            Q += self.arrays_diff(arr1, arr2) 
        return Q

    def more_learners_difference_on_sentence(self, sent_index):
        Q = 0
        for learner_a, learner_b in zip(self.learners_predictions[0::1], self.learners_predictions[1::1]):
            Q += self.sentence_difference(learner_a[sent_index], learner_b[sent_index])
        return Q

    def more_learners_difference_on_all_sentences(self):
        self.Q = {}
        for sentence in range(0, self.num_sentences):
            self.Q[sentence] = self.more_learners_difference_on_sentence(sentence)
        return self.Q

    def save_Q(self, name):
        print("[INFO] Saving Q object to '" + name + "'\n")
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(self.Q, f, protocol = pickle.HIGHEST_PROTOCOL)

    def load_Q(self, name):
        print("[INFO] Loading Q object from '" + name + "'\n")
        with open(name + '.pkl', 'rb') as f:
            self.Q = pickle.load(f)

    def get_new_train_set_indices(self, threshold):
        return [key for key, val in self.Q.items() if val >= threshold]
        
