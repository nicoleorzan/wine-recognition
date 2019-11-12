import re
import nltk.stem as stem
from nltk import pos_tag


class Sentence_Tagger:

    def __init__(self, words, sentences):
        self.words = words
        self.sentences = sentences
        #self.st = stem.RegexpStemmer('ing$|s$|y$|ly$|ed$', min=4)
        #self.regex = re.compile('[^a-zA-Z]')

    def sentence_setup(self, sent):
        
        '''preparing tuplas of sentences where the pos tag is already 
        performed, but we still have to perfrom the BIO tag. 
        Tuple form: ('word', 'verb/noun/whatever', ''O'') '''
        
        lenw = 0
        tupla = []
        POS_tag = pos_tag(sent.split())
        for idx, w in enumerate(sent.split()): 
            tupla.append((w, lenw, lenw+len(w), POS_tag[idx][1], ''O'', 0))
            lenw += len(w)+1
        return tupla


    # ==============================================================
    # ==============================================================

    def BIO_tag(self, label, list_of_words, sent, tupla):
        
        '''function performing BIO tag inside the tuplas'''
        
        regex = re.compile('[^a-zA-Z]')
        st = stem.RegexpStemmer('ing$|s$|y$|ly$|ed$', min=4)
        
        lenw=0
        for g in list_of_words:
            if g in sent:
                lis = g.split()
                
                if len(lis)>1: # my lis is composed of more words
                    #print(lis)
                    for idxx, word in enumerate(lis):
                        word = regex.sub('', word)
                        word = st.stem(word)
                        # cerco la prima parola della lista nelle tuple della frase, 
                        # cercando non per parola ma per indice della parola e lunghezza della stessa
                        for idx, t in enumerate(tupla):
                            if t[1]==sent.index(g)+lenw:
                                if t[5]==0:
                                    if idxx == 0:
                                        tupla[idx] = ((t[0], sent.index(g)+lenw, sent.index(g)+lenw + len(word), t[3],'B'+'-'+label, 1))
                                    else:
                                        tupla[idx] = ((t[0], sent.index(g)+lenw, sent.index(g)+lenw + len(word), t[3], 'I'+'-'+label, 1))
                        lenw+=len(word)+1
                        
                else: # cerco cose composte da una sola parola
                    for idx, t in enumerate(tupla):
                            if regex.sub('', st.stem(t[0])) == g and t[1]==sent.index(g)+lenw:
                                if t[5]==0:
                                    if (idx!=len(tupla)-1 and (tupla[idx+1][3] == 'JJ')):# or tupla[idx+1][3] == 'NN') ):
                                        tupla[idx] = ((t[0], sent.index(g)+lenw, sent.index(g)+lenw + len(g), t[3], 'B'+'-'+label, 1))
                                        tupla[idx+1] = ((tupla[idx+1][0], tupla[idx+1][1], tupla[idx+1][2], tupla[idx+1][3], 'I'+'-'+ label, 1))
                                                                            
                                    elif (idx!=0 and ("," not in tupla[idx-1][3]) and (tupla[idx-1][3] == 'JJ')):# or tupla[idx-1][3] == 'NN') ):
                                        tupla[idx-1] = ((tupla[idx-1][0], tupla[idx-1][1], tupla[idx-1][2], tupla[idx-1][3], 'B'+'-'+ label, 1))
                                        tupla[idx] = ((t[0], sent.index(g)+lenw, sent.index(g)+lenw + len(g), t[3], 'I'+'-'+label, 1))
                                                
                                    else:
                                        tupla[idx] = ((t[0], sent.index(g)+lenw, sent.index(g)+lenw + len(g), t[3], 'B'+'-'+label, 1))

        return(tupla)

    # ============================================================================
    # ============================================================================


    def labeling_function(self, sentences, words_tag_dict):
        
        ''' MAIN labeing function which calls sentence_setup and BIO_tag '''
        
        tag_list = []
        for i, sent in enumerate(sentences):
            if (i%50 == 0): print(i)
            sent = sent.lower()
            #sent = sent.translate(str.maketrans('', '', string.punctuation))
            sent = sent.replace(",", "").replace(".", "")
            tag_tup = self.sentence_setup(sent)
            
            
            for key, val in words_tag_dict.items():
                tag_tup = self.BIO_tag(key, val, sent, tag_tup)

            tag_list.append(tag_tup)
            
            
        return (tag_list)

    def cleaning(self, tupla_list):
        
        '''eliminating useless informatons from tuplas'''
        
        tupla_list_cleaned = []
        for tt in tupla_list:
            tmp = []
            for t in tt:
                tmp.append((t[0], t[3], t[4]))
            tupla_list_cleaned.append(tmp)
        
        return(tupla_list_cleaned)