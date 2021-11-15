#import datetime
#import base64
import numpy as np
import os
import json
import spacy
import pandas as pd

from aiohttp import web

from wildhack_modules.pipeline import load_numpy_df
from wildhack_modules.lib_gensim import fasttext_gensim_loadmodule, gensim_fasttext_get_sentence_vector
from wildhack_modules.pipeline import vektors_model_fasttext, vektors_fasttext_vectorize
from wildhack_modules.pipeline import create_common_futures_learn
from wildhack_modules.lib_pandas import create_pairwise_dataframe_reverse
from wildhack_modules.pipeline import create_common_futures_answer

from wildhack_modules.lib_faiss import faiss_create_index, faiss_search, faiss_normalize_L2

#from card_find_contours import find_cardinfo_on_image
#from cards_const import global_max_length_recive_file, global_empty_card_info, global_tmp_dir


#def write_file(body, filename):
    ## for i in range(80000000):
    ##     a = a * 2 / 2 + 1 * 2
    #file_handle = open(filename, "wb")
    #file_handle.write(body)

df_query_popularity = []
np_query_popularity = np.zeros(0)
model_nlp = 0
model_vectors = 0
faiss_index_query_popularity = 0

df_ktru_okpd2 = []
np_ktru_okpd2 = np.zeros(0)
faiss_index_ktru_okpd2 = 0

def load_vectors():
    global df_query_popularity, faiss_index_query_popularity
    
    global df_ktru_okpd2, faiss_index_ktru_okpd2
    
    df_query_popularity, np_query_popularity = load_numpy_df('query_popularity_ktru')

    df_ktru_okpd2, np_ktru_okpd2 = load_numpy_df('ktru_okpd2')

    faiss_index_query_popularity = faiss_create_index(300)
    
    faiss_normalize_L2(np_query_popularity)
    faiss_index_query_popularity.add(np_query_popularity)

    print(np_query_popularity.shape)

    faiss_index_ktru_okpd2 = faiss_create_index(300)
    
    faiss_normalize_L2(np_ktru_okpd2)
    faiss_index_ktru_okpd2.add(np_ktru_okpd2)

    
def load_models():    
    global model_nlp, model_vectors
    model_nlp = spacy.load("ru_core_news_md")
    model_vectors = fasttext_gensim_loadmodule('models/fasttext/wiki.ru_gensim.model')
    
def create_common_futures_lemma(df, name_column):
    '''
    Предобработка колонки learn, для обучения
    '''
    global model_nlp
    
    df['lemma'] = df[name_column].apply(lambda x: [(token.lemma_) for token in model_nlp(x)]).str.join(' ')

async def get_root(request):
    text = "Good"
    return web.Response(text=text)

def ranking_algorithm_fasttext_faiss(faiss_index, vectors_search, topn=10):
    '''
    Ранжирование с помощью faiss
    '''
    faiss_normalize_L2(vectors_search)
    
    sort_distance, sort_indexs = faiss_search(faiss_index, vectors_search, topn)
    
    return sort_distance, sort_indexs

def dict_to_json(dict_str):
    return json.dumps(dict_str, ensure_ascii=False)

def view_category(df, np_vectors):
    '''
    Функция возвращает к какой категории сайта подходит
    '''
    global np_ktru_okpd2, df_ktru_okpd2
    
    sort_distance, sort_indexs = ranking_algorithm_fasttext_faiss(faiss_index_ktru_okpd2, np_vectors, topn=1)
    df_result = create_pairwise_dataframe_reverse(df, df_ktru_okpd2, sort_indexs, sort_distance) 
    
    if len(df_result) > 0:     
        categorie = df_result['kode'][0]
    else:
        categorie = ''
        
    return categorie

def result_df(quary_text):
    
    global model_vectors
    
    print('== 01 ==')
    
    df = pd.DataFrame([quary_text],columns=['search'])
    
    create_common_futures_learn(df, 'search')
    
    create_common_futures_lemma(df, 'learn')
    
    print('== 02 ==')
    
    np_vectors = vektors_fasttext_vectorize(model_vectors, df['lemma'])
    
    category_query = view_category(df, np_vectors)
    
    topn = 200
    
    sort_distance, sort_indexs = ranking_algorithm_fasttext_faiss(faiss_index_query_popularity, np_vectors, topn=topn)

    print('== 04 ==')
    
    df_result = create_pairwise_dataframe_reverse(df, df_query_popularity, sort_indexs, sort_distance) 
    
    df_result = sub_quary(df_result)
    
    create_common_futures_answer(df_result, 'target_query')
    
    return df_result, category_query

def sub_quary(df_result):
    
    #Филтруем
    scale = 1.2
    
    df_result = df_result[df_result['target_shore'] < 1]
    df_result.loc[(df_result.ves == 5), 'target_query_popularity'] *= scale
    df_result['target_query_popularity'] *= 10
    df_result = df_result[df_result['target_query_popularity'] > 0]
    df_result['target_shore'] = df_result['target_shore'] * df_result['target_query_popularity'] / 100
    df_result = df_result.sort_values('target_shore', ascending=False)
    df_result = df_result.drop_duplicates(subset=['kode','ktru'],keep='first')
    
    df_result1 = df_result[:5]

    print('== 05 ==')
    
    if len(df_result) > 5:
        n_sample = min(len(df_result)-5,5)
        df_result2 = df_result[5:].sample(n_sample)
    
        df_result = df_result1.append(df_result2)
    else:
        df_result = df_result1

    print('== 05 ==')
    
    return df_result

async def sendbase(request):
    
    global df_query_popularity, np_query_popularity

    #if request.content_length > global_max_length_recive_file:
        #return web.Response(text="Ok")

    if request.body_exists:
        #json_data = await request.json()
        try:                
            json_data = await request.json()
        except: 
            print("Ошибочный запрос") 
            return web.Response(status=205, text='Error')
                    
    if 'search' not in json_data.keys():
        return web.Response(status=204)
    
    target_query = ['тэг 1','тэг 2', 'тэг 3','тэг 4','тэг 5','тэг 6','тэг 7','тэг 8','тэг 9','тэг 10']

    quary_text = json_data['search']
    
    df_result, category_query = result_df(quary_text)

    target_query = df_result['answer'].values
    #target_query = df_result['target_query'].values
    target_query = target_query[:10]
    target_query = list(target_query)

    print('== 06 ==')

    
    print('New quary...')
    print(json_data['search'])
    print('Result:', target_query)


    tag_info = {'search': json_data['search'],
                'okpd': category_query,
                'tags': target_query,
        'result': 'Хорошо'}

    #ret = web.json_response(tag_info, content_type="application/json;charset=utf-8")

    print('== 07 ==')

    
    t_json = dict_to_json(tag_info)

    print('== 08 ==')

    
    ret = web.json_response(text=t_json)
    
    #d = ret.headers.get('content-type')
    #ret._headers = {"Content-Type": "application/json;charset=utf-8"}

    return ret


app = web.Application() #client_max_size=global_max_length_recive_file
app.add_routes([web.get('/', get_root),
                web.post('/api/get_tags', sendbase)])

if __name__ == '__main__':
    
    print('Starting...')
    
    load_vectors()
    load_models()
    
    server_host = os.environ.get('recognition_server_host')
    server_port = os.environ.get('recognition_server_port')

    if server_host is None:
        server_host = '0.0.0.0'

    if server_port is None:
        server_port = 5000

    print('Run web server...')

    web.run_app(app, host=server_host, port=server_port)
