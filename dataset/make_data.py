import pandas as pd
import argparse
import json
import os
import re


# label  {'chung' : 0, 'junla' :1, 'kyung' :2, 'zeju' : 3}

parser = argparse.ArgumentParser(description='Transformer dialect machine translation')
parser.add_argument('--data', default='/nas/home/sungchul/dia/chung_train/val',type=str,
                    help='path to dataset directory')



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
        if len(set(df['standard_form'][i].split(' ')) - set(df['dialect_form'][i].split(' '))) > 1 : 
            filter_li.append(i)
                
    df2 = df2.loc[filter_li]
    df2 = df2.reset_index(drop=True)
    
    hangul = re.compile('[^ ㄱ-ㅣ가-힣+]')
    df2['standard_form'] = df2['standard_form'].apply(lambda x : hangul.sub('',x))
    df2['dialect_form'] = df2['dialect_form'].apply(lambda x : hangul.sub('',x))

    special = re.compile(r'[^ A-Za-z0-9가-힣+]')
    df2['standard_form'] = df2['standard_form'].apply(lambda x : special.sub('',x))
    df2['dialect_form'] = df2['dialect_form'].apply(lambda x : special.sub('',x))

    tr = train = df2.iloc[5000:]
    val = df2.iloc[0:2500]
    test = df2.iloc[2500:5000]

    return tr, val, test

if __name__ == '__main__':
    args = parser.parse_args() 
    path = args.data

    tr, val, test = preprocess(path)    

    tr.to_csv('train.csv')
    val.to_csv('val.csv')
    test.to_csv('test.csv')
