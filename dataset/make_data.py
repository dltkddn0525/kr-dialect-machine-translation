def make_data(data_c, data_j, data_k, data_z): 
    data_c = pd.read_csv(data_c, index_col=0)
    data_j = pd.read_csv(data_j, index_col=0)
    data_k = pd.read_csv(data_k, index_col=0)
    data_z = pd.read_csv(data_z, index_col=0)

    data = pd.concat([data_c, data_j, data_k, data_z],axis=0)
    data = data.reset_index(drop=True)
     
    hangul = re.compile('[^ ㄱ-ㅣ가-힣+]')
    data['standard_form'] = data['standard_form'].apply(lambda x : hangul.sub('',x))
    data['dialect_form'] = data['dialect_form'].apply(lambda x : hangul.sub('',x))

    special = re.compile(r'[^ A-Za-z0-9가-힣+]')
    data['standard_form'] = data['standard_form'].apply(lambda x : special.sub('',x))
    data['dialect_form'] = data['dialect_form'].apply(lambda x : special.sub('',x))
    
    return data 


data1 = ' '
data2 = ' '
data3 = ' '
data4 = ' ' 

data  = make_data(data1, data2, data3, data4)
data.to_csv('train.csv')
