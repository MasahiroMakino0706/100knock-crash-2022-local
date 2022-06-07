import numpy as np
import pandas as pd
import torch
import torch.nn as nn


#データの読み取り関数
def FromcsvTotensor(file):
    result = np.loadtxt(file + '.csv')
    result = torch.from_numpy(result.astype(np.float32)) 
    return result

#X_trainの読み取り
X_train = FromcsvTotensor('X_train')

#活性化関数の定義
activation = nn.Softmax(dim=0)

#ニューラルネットワーク構築
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(300, 4, bias=False)
        
    def forward(self, input):
        output = activation(self.layer1(input))
        return output
    
model = NeuralNetwork()

#Yの生成
result = []
for i in range(4):
    result.append(model(X_train[i]))

#Yのプリント
print(result)