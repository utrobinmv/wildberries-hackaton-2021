#import datetime
#import base64
import numpy as np
import os
import json
import spacy
import pandas as pd

from flask import Flask, jsonify, abort
from flask import request
from flask import make_response

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

from wildhack_modules.pipeline import load_numpy_df
from wildhack_modules.lib_gensim import fasttext_gensim_loadmodule, gensim_fasttext_get_sentence_vector
from wildhack_modules.pipeline import vektors_model_fasttext, vektors_fasttext_vectorize
from wildhack_modules.pipeline import create_common_futures_learn
from wildhack_modules.pipeline import create_common_futures_answer
from wildhack_modules.lib_pandas import create_pairwise_dataframe_reverse

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
faiss_index = 0

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/')
def index():
    return "Energy predict"



def load_vectors():
    global df_query_popularity, faiss_index
    df_query_popularity, np_query_popularity = load_numpy_df('query_popularity_ktru')

    faiss_index = faiss_create_index(300)
    
    faiss_normalize_L2(np_query_popularity)
    faiss_index.add(np_query_popularity)

    print(np_query_popularity.shape)
    
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

#def detect_card(body, name_request):
    #card_info = global_empty_card_info

    #img_array = np.asarray(bytearray(body), dtype=np.uint8)
    #image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    #time_str = str(datetime.datetime.now()).replace(" ","_").replace(":","-")

    #file_name = name_request + "-" + time_str

    #file_name = global_tmp_dir + file_name + ".jpg"

    #write_file(body, file_name)

    #print("request: ", file_name)

    #(max_y, max_x, color_channel) = image.shape
    #if color_channel == 3:
        #card_info = find_cardinfo_on_image(image, time_str)

    #print("return: ", card_info)

    #return web.json_response(card_info)

#async def sendphoto(request):

    #if request.content_length > global_max_length_recive_file:
        #return web.Response(text="Ok")

    #if request.body_exists:
        #body = await request.read()
    #else:
        #return web.Response(text="Ok")

    #return detect_card(body, "sendphoto")

def ranking_algorithm_fasttext_faiss(vectors, vectors_search, topn=10):
    '''
    Ранжирование с помощью faiss
    '''
    global faiss_index
    
    faiss_normalize_L2(vectors_search)
    
    sort_distance, sort_indexs = faiss_search(faiss_index, vectors_search, topn)
    
    return sort_distance, sort_indexs

def dict_to_json(dict_str):
    return json.dumps(dict_str, ensure_ascii=False)


@app.route('/api/get_tags', methods=['POST'])
def get_tags():
    global df_query_popularity, np_query_popularity
    global model_vectors

    if not request.json and "USE_FACT" in request.json:
        abort(400)
    print(request.json)

    json_data = request.json

    if 'search' not in json_data.keys():
        return web.Response(status=204)
    
    #result = ['тэг 1','тэг 2', 'тэг 3','тэг 4','тэг 5','тэг 6','тэг 7','тэг 8','тэг 9','тэг 10']

    quary_text = json_data['search']
    
    df = pd.DataFrame([quary_text],columns=['search'])
    
    create_common_futures_learn(df, 'search')
    
    create_common_futures_lemma(df, 'learn')
    
    np_vectors = vektors_fasttext_vectorize(model_vectors, df['lemma'])
    
    topn = 200
    
    sort_distance, sort_indexs = ranking_algorithm_fasttext_faiss(np_query_popularity, np_vectors, topn=topn)
    
    df_result = create_pairwise_dataframe_reverse(df, df_query_popularity, sort_indexs, sort_distance) 
    df_result = df_result.drop_duplicates(subset=['kode','ktru'],keep='first')
    
    df_result1 = df_result[:5]
    
    if len(df_result) > 5:
        n_sample = min(len(df_result)-5,5)
        df_result2 = df_result[5:].sample(n_sample)
    
        df_result = df_result1.append(df_result2)
    else:
        df_result = df_result1
        
    create_common_futures_answer(df_result, 'target_query')
    
    target_query = df_result['answer'].values
    #target_query = df_result['target_query'].values
    target_query = target_query[:10]
    target_query = list(target_query)
    
    print('New quary...')
    print(json_data['search'])
    print('Result:', target_query)

    tag_info = {'search': json_data['search'],
                'tags': target_query,
        'result': 'Хорошо'}

    #ret = web.json_response(tag_info, content_type="application/json;charset=utf-8")
    
    t_json = dict_to_json(tag_info)
    
    #ret = web.json_response(text=t_json)
    
    #d = ret.headers.get('content-type')
    #ret._headers = {"Content-Type": "application/json;charset=utf-8"}

    return jsonify(t_json), 201

#async def sendbase(request):
    #global df_query_popularity, np_query_popularity
    #global model_vectors

    ##if request.content_length > global_max_length_recive_file:
        ##return web.Response(text="Ok")

    #if request.body_exists:
        ##json_data = await request.json()
        #try:                
            #json_data = await request.json()
        #except: 
            #print("Ошибочный запрос") 
            #return web.Response(status=205, text='Error')
                
    ##else:
        ##return web.Response(text="Ok")

    ##raw_image = json_data['image']

    ##body = base64.b64decode(raw_image)

    ##return detect_card(body, "sendbase")
    
    #if 'search' not in json_data.keys():
        #return web.Response(status=204)
    
    ##result = ['тэг 1','тэг 2', 'тэг 3','тэг 4','тэг 5','тэг 6','тэг 7','тэг 8','тэг 9','тэг 10']

    #quary_text = json_data['search']
    
    #df = pd.DataFrame([quary_text],columns=['search'])
    
    #create_common_futures_learn(df, 'search')
    
    #create_common_futures_lemma(df, 'learn')
    
    #np_vectors = vektors_fasttext_vectorize(model_vectors, df['lemma'])
    
    #topn = 200
    
    #sort_distance, sort_indexs = ranking_algorithm_fasttext_faiss(np_query_popularity, np_vectors, topn=topn)
    
    #df_result = create_pairwise_dataframe_reverse(df, df_query_popularity, sort_indexs, sort_distance) 
    #df_result = df_result.drop_duplicates(subset=['kode','ktru'],keep='first')
    
    #df_result1 = df_result[:5]
    
    #df_result2 = df_result[5:].sample(5)
    
    #df_result = df_result1.append(df_result2)
    
    #target_query = df_result['target_query'].values
    #target_query = target_query[:10]
    #target_query = list(target_query)
    
    #print('New quary...')
    #print(json_data['search'])
    #print('Result:', target_query)

    #tag_info = {'search': json_data['search'],
                #'tags': target_query,
        #'result': 'Хорошо'}

    ##ret = web.json_response(tag_info, content_type="application/json;charset=utf-8")
    
    #t_json = dict_to_json(tag_info)
    
    #ret = web.json_response(text=t_json)
    
    ##d = ret.headers.get('content-type')
    ##ret._headers = {"Content-Type": "application/json;charset=utf-8"}

    #return ret


#app = web.Application() #client_max_size=global_max_length_recive_file
#app.add_routes([web.get('/', get_root),
                #web.post('/api/get_tags', sendbase)])

if __name__ == '__main__':
    
    print('Starting...')
    
    load_vectors()
    load_models()
    
    server_host = os.environ.get('recognition_server_host')
    server_port = os.environ.get('recognition_server_port')

    print('Run web server...')
    app.run(debug=False, host='0.0.0.0', port=5000)

    #if server_host is None:
        #server_host = '0.0.0.0'

    #if server_port is None:
        #server_port = 5000


    #web.run_app(app, host=server_host, port=server_port)
