import pandas as pd

def create_pairwise_dataframe(df, de, np_result, np_shore, columns_add = 'target', column_shore = 'target_shore'):
    '''
    Создание DataFrame из результатов поиска faiss
    '''
    
    de_result = de.iloc[:0].copy()
    
    df_copy = df.reset_index()
    if 'index' in df.columns:
        raise ValueError ( "В исходном DataFrame не должно быть колонки index!" )        
    if 'index' in de.columns:
        raise ValueError ( "В исходном DataFrame не должно быть колонки index!" )        

    df_columns = ['index']
    for old in df_columns:
        de_result[columns_add + '_' + old] = ''
        
    #print('do copy',df_copy.shape, df_copy.columns)
    df_copy.columns = [columns_add + '_' + column for column in df_copy.columns]
    #df_columns = [columns_add + '_' + column for column in df_columns]
    #print('posle copy',df_copy.shape, df_copy.columns)
    
    index_col_name = columns_add + '_' + df_columns[0]
    df_copy_index_col = df_copy.columns.tolist().index(index_col_name)
    
    de_result[column_shore] = 0
    
    list_de = []
        
    for idx_de in range(np_result.shape[0]):
        for j in range(np_result.shape[1]):
            idx_df = np_result[idx_de,j]

            shore = np_shore[idx_de,j]
        
            if shore > 0:
                de_series = de.iloc[idx_de].copy()
                
                de_series[index_col_name] = df_copy.iloc[idx_df,df_copy_index_col]
                de_series[column_shore] = shore
        
                list_de.append(de_series)
    
    de_result = pd.DataFrame(list_de)
    
    #print('do',de_result.shape, de_result.columns)
    
    de_result = de_result.merge(df_copy, how='left', left_on='target_index', right_on='target_index')
    
    #print('posle',de_result.shape, de_result.columns)
    
    de_result.drop('target_index', inplace=True, axis=1)
    
    de_result = de_result.reset_index(drop=True)
    
    return de_result

def create_pairwise_dataframe_reverse(df_toledo, de_supplier, sort_indexs, sort_distance, columns_add = 'target', column_shore = 'target_shore'):
    '''
    Создание DataFrame из результатов поиска faiss
    '''
    
    de_result = de_supplier.iloc[:0].copy()
    
    df_copy = df_toledo.reset_index()
    if 'index' in df_toledo.columns:
        raise ValueError ( "В исходном DataFrame не должно быть колонки index!" )        
    if 'index' in de_supplier.columns:
        raise ValueError ( "В исходном DataFrame не должно быть колонки index!" )        

    df_columns = ['index']
    for old in df_columns:
        de_result[columns_add + '_' + old] = ''
        
    #print('do copy',df_copy.shape, df_copy.columns)
    df_copy.columns = [columns_add + '_' + column for column in df_copy.columns]
    #df_columns = [columns_add + '_' + column for column in df_columns]
    #print('posle copy',df_copy.shape, df_copy.columns)
    
    index_col_name = columns_add + '_' + df_columns[0]
    df_copy_index_col = df_copy.columns.tolist().index(index_col_name)
    
    de_result[column_shore] = 0
    
    list_de = []
        
    for idx_df in range(sort_indexs.shape[0]):
        for j in range(sort_indexs.shape[1]):
            idx_de = sort_indexs[idx_df,j]

            shore = sort_distance[idx_df,j]
        
            #if shore > 0:
            de_series = de_supplier.iloc[idx_de].copy()

            de_series[index_col_name] = df_copy.iloc[idx_df,df_copy_index_col]
            de_series[column_shore] = shore

            list_de.append(de_series)
    
    de_result = pd.DataFrame(list_de)
    
    #print('do',de_result.shape, de_result.columns)
    
    de_result = de_result.merge(df_copy, how='left', left_on='target_index', right_on='target_index')
    
    #print('posle',de_result.shape, de_result.columns)
    
    de_result.drop('target_index', inplace=True, axis=1)
    
    de_result = de_result.reset_index(drop=True)
    
    return de_result