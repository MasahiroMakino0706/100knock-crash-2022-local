import torch
import torch.nn as nn
import numpy as np
'''
from Q71 import FromcsvTotensor

関数のみではなくQ71に記述されたプログラム全てが呼び出される?ため無駄な出力が出る
具体的には関数読み込みだけでなくQ71最後のプリント行も出力される
→
Q71の関数をQ72にコピぺする
関数定義のみを行う別ファイルを作り、そのファイルから関数をインポートする
'''

#ニューラルネットワーク構築
activation = nn.Softmax(dim=0)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(300, 4, bias=False)
        
    def forward(self, input):
        output = activation(self.layer1(input))
        return output
    
model = NeuralNetwork()

#データの取得
def X_FromcsvTotensor(file):
    result = np.loadtxt(file + '.csv')
    result = torch.from_numpy(result.astype(np.float32)) 
    return result

def Y_FromcsvTotensor(file):
    result = np.loadtxt(file + '.csv')
    result = torch.from_numpy(result.astype(np.int64)) 
    return result

X_train = X_FromcsvTotensor('X_train')
Y_train = Y_FromcsvTotensor('Y_train')

#損失を計算する関数の定義
def make_entropy(train, label, i):
    entropy = -model(train[i])[label[i]].log()
    return entropy

loss = nn.CrossEntropyLoss()
loss = loss(model(X_train[0]), Y_train[0])#long型にする必要あり
model.zero_grad()
loss.backward()
print('x1の計算結果')
print(f'損失：{loss}')
print(f'勾配：{model.layer1.weight.grad}')

loss = nn.CrossEntropyLoss()
loss = loss(model(X_train[:4]), Y_train[:4])#long型にする必要あり
model.zero_grad()
loss.backward()
print('\n事例集合に対する計算結果')
print(f'損失：{loss}')
print(f'勾配：{model.layer1.weight.grad}')

print('\n損失が正しく計算されているか確認')
loss1 = nn.CrossEntropyLoss()
print((loss1(model(X_train[0]), Y_train[0])+ loss1(model(X_train[1]), Y_train[1]) + loss1(model(X_train[2]), Y_train[2]) + loss1(model(X_train[3]), Y_train[3]))/4)




