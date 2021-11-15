import faiss

def faiss_create_index(size=100):
    '''
    Создание индекса Faiss
    '''
    index = faiss.index_factory(size, "Flat", faiss.METRIC_INNER_PRODUCT)
    return index


def faiss_normalize_L2(output_ids_df):
    '''
    Нормализует вектора
    '''
    faiss.normalize_L2(output_ids_df)

def faiss_search(index, vectors, topn):
    '''
    Запуск поиска Faiss
    '''
    return index.search(vectors, topn)
    
    