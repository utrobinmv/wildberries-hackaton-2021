import numpy as np

def get_embedings_from_list(text, model, embeddings_function):
    '''
    Функция создает список векторов в виде numpy массива
    '''
    vectors = np.zeros(0)
    for idx, sent in enumerate(text):
        vect = embeddings_function(sent, model)
        vect = vect[np.newaxis, :]
        if idx == 0:
            vectors = vect
        else:
            vectors = np.concatenate([vectors, vect])

    return vectors

