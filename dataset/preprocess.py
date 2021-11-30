import pandas as pd 

def preprocess(path):
    file_list = os.listdir(path)

    file_list = [file for file in file_list if file.endswith('json')]
    file_list


    li = []
    li2 = []
    for i in file_list :
        print(i)
        with open(os.path.join(path,i), 'rb') as f: 
            df = json.load(f)
        kk = [i['standard_form'] for i in df['utterance']]     
        kk2 = [i['dialect_form'] for i in df['utterance']]
        for i in kk:
            li.append(i)
        for j in kk2:
            li2.append(j)    

    df = pd.DataFrame({'standard_form' : li,
                       'dialect_form' : li2,
                       'label' : 0})

    df2 = df[df['standard_form'] != df['dialect_form']]

    filter_li = []
    for i in df2.index : 
        if len(set(df['standard_form'][i].split(' ')) - set(df['dialect_form'][i].split(' '))) > 2 : 
            filter_li.append(i)
                
    df2 = df2.loc[filter_li]
    df2 = df2.reset_index(drop=True)


    tr = df2.loc[:df2.shape[0]*0.8]
    val = df2.loc[df2.shape[0]*0.8:]

    return tr, val


path = ' '

tr, val = preprocess(path)    

tr.to_csv('train.csv')
val.to_csv('val.csv')
