import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import gensim
from gensim.models import word2vec
from gensim.models import KeyedVectors

#データの取り出し
train = pd.read_table('./train.txt')
valid = pd.read_table('./valid.txt')
test = pd.read_table('./test.txt')

#タイトル列のみの抽出→特徴量の生成
X_train = train['TITLE']
X_valid = valid['TITLE']
X_test = test['TITLE']

#カテゴリ列のみの抽出→ラベルの生成
Y_train = train['CATEGORY']
Y_valid = valid['CATEGORY']
Y_test = test['CATEGORY']

#scpコマンドでカレントディレクトリにファイルを置いた後word2vecをダウンロード
EMBEDDING_FILE = './GoogleNews-vectors-negative300.bin.gz'
word_vectors = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

#ベクトル表現の生成関数を定義
def create_vector(data):
    result = []
    for i in range(len(data)):
        listdata = data[i].split(' ')
        subresult = np.zeros(300)
        number = 0
        for j in range(len(listdata)):
            try:
                subresult += word_vectors[listdata[j]]
                number += 1
            except Exception as e: #単語埋め込み範囲外の語彙を見つけた時に例外処理
                pass
        if number == 0:
            result.append(subresult)
        else:
            result.append(subresult/number)
    return result

#特徴量においてベクトル表現を生成
X_train = create_vector(X_train)
X_valid = create_vector(X_valid)
X_test = create_vector(X_test)

#ラベルにおいて特徴量を生成
dct_int = {'b': 0, 't': 1, 'e': 2,'m': 3}
Y_train = Y_train.replace(dct_int)
Y_valid = Y_valid.replace(dct_int)
Y_test = Y_test.replace(dct_int)

#型を合わせる
X_train = torch.tensor(np.array(X_train)).float()
X_valid = torch.tensor(np.array(X_valid)).float()
X_test = torch.tensor(np.array(X_test)).float()

Y_train = torch.tensor(Y_train.values)
Y_valid = torch.tensor(Y_valid.values)
Y_test = torch.tensor(Y_test.values)

#tensor->numpy->DataFrame->csv変換関数
def tensorTocsv(tenso, file):
    np.savetxt(file, tenso.numpy())

#関数適用
tensorTocsv(X_train, 'X_train.csv')
tensorTocsv(X_valid, 'X_valid.csv')
tensorTocsv(X_test, 'X_test.csv')
tensorTocsv(Y_train, 'Y_train.csv')
tensorTocsv(Y_valid, 'Y_valid.csv')
tensorTocsv(Y_test, 'Y_test.csv')

"confirm"