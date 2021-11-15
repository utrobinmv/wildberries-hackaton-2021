import numpy as np

import gensim

#FastText
def fasttext_gensim_loadmodule(modulename):
    '''
    Загружает модель FastText
    '''
    ft = gensim.models.KeyedVectors.load(modulename)
    return ft

def gensim_fasttext_get_sentence_vector(sent, model):
    '''
    Функция конвертирует предложение в вектор FastText
    '''
    
    vector_words = []
    
    list_words = sent.split(' ')
    for word in list_words:
        vector_words.append(model.get_vector(word, norm=True))
            
    if len(vector_words) > 0:
        vector_words = np.array(vector_words)
        vector_words = np.mean(vector_words,axis=0)
    else:
        vector_words = np.zeros(model.vector_size)
    
    return vector_words
