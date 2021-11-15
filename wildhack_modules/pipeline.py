from .lib_gensim import fasttext_gensim_loadmodule, gensim_fasttext_get_sentence_vector
from .lib_numpy import get_embedings_from_list
from .lib_faiss import faiss_create_index, faiss_search, faiss_normalize_L2
from .lib_re import tokenize_corpus

import spacy
import pickle

def create_learn_text(text,tokenizer_kwargs):
    tokenized_text = tokenize_corpus(text,**tokenizer_kwargs)
    
    new_text = []
    for next_list in tokenized_text:
        new_str = ' '.join(next_list)
        new_text.append(new_str)
    
    return new_text

def create_common_futures_answer(df, name_column):
    '''
    Предобработка колонки learn, для обучения
    '''
    
    #print('Start tr_create_common_futures_learn',len(df))
    
    df['answer'] = create_learn_text(df[name_column].values, {'min_token_size':0, 'not_lower': True})

def create_common_futures_learn(df, name_column):
    '''
    Предобработка колонки learn, для обучения
    '''
    
    #print('Start tr_create_common_futures_learn',len(df))
    
    df['learn'] = create_learn_text(df[name_column].values, {'min_token_size':0})
    
def create_common_futures_lemma(df, name_column):
    '''
    Предобработка колонки learn, для обучения
    '''
    
    nlp = spacy.load("ru_core_news_md")
    
    df['lemma'] = df[name_column].apply(lambda x: [(token.lemma_) for token in nlp(x)]).str.join(' ')

def create_common_futures_lemma_tag(df, name_column):
    '''
    Предобработка колонки learn, для обучения
    '''
    
    nlp = spacy.load("ru_core_news_md")
    
    df['pos'] = df[name_column].apply(lambda x: [(token.pos_) for token in nlp(x)]).str.join(' ')

def vektors_model_fasttext():
    '''
    Создание и обучение модели fasttext
    '''

    model = None
    model = fasttext_gensim_loadmodule('models/fasttext/wiki.ru_gensim.model')
    #model = fasttext_gensim_loadmodule('models/fasttext/shrinked_fasttext.model')

    return model

def vektors_fasttext_vectorize(model, series_df):
    
    result = get_embedings_from_list(series_df.values, model, gensim_fasttext_get_sentence_vector)

    return result

def ranking_algorithm_fasttext_faiss(vectors, vectors_search, topn=10):
    '''
    Ранжирование с помощью faiss
    '''
    index = faiss_create_index(300)
    
    faiss_normalize_L2(vectors)
    index.add(vectors)
    
    faiss_normalize_L2(vectors_search)
    
    sort_distance, sort_indexs = faiss_search(index, vectors_search, topn)
    
    return sort_distance, sort_indexs

def save_numpy_df(df,np_vectors,name):
    folder = 'result/'
    pickle.dump(df, open(folder+name+'_df.pkl', 'wb'))
    pickle.dump(np_vectors, open(folder+name+'_vectors.pkl', 'wb'))
    
def load_numpy_df(name):
    folder = 'result/'
    df = pickle.load(open(folder+name+'_df.pkl', 'rb'))
    np_vectors = pickle.load(open(folder+name+'_vectors.pkl', 'rb'))
    return df, np_vectors