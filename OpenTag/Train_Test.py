from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class Train_Test:
    
    def __init__(self, tagged_sentences, tags, words):
        self.tagged_sentences = tagged_sentences
        self.tags = tags
        self.words = words
        self.n_tags = len(tags)
        self.n_words = len(words)
        self.max_len = len(max(self.tagged_sentences, key=len))

        self.word2idx = {w: i + 1 for i, w in enumerate(self.words)}
        self.tag2idx = {t: i for i, t in enumerate(self.tags)}
        self.idx2tag = {i: w for w, i in self.tag2idx.items()}

        self.indexing()
        
    def get_number_words(self):
        return self.n_words
    
    def get_number_tags(self):
        return self.n_tags

    def get_max_len(self):
        return self.max_len
        
    def padding(self, seq, val):
        return pad_sequences(maxlen = self.max_len, sequences = seq, padding="post", value = val)
        
    def indexing(self):
        X = [[self.word2idx[w[0]] for w in s] for s in self.tagged_sentences]
        y = [[self.tag2idx[w[2]] for w in s] for s in self.tagged_sentences]
        
        self.X = self.padding(X, self.n_words)  
        y = self.padding(y, self.tag2idx["O"])

        self.y = to_categorical(y, num_classes = self.n_tags)

    def get_x_y(self):
        return (self.X, self.y)
    
    def train_test(self, test_size, r_stat=123):
        return train_test_split(self.X, self.y, test_size=test_size, random_state=r_stat)
     