import torch
import torch.nn as nn
import numpy as np

#ニューラルネットワーク構築
activation = nn.Softmax(dim=0)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(300, 4, bias=False)
        
    def forward(self, input):
        output = self.layer1(input) #loss_entropy = nn.CrossEntropyLoss()上で与える予測値はソフトマックス関数により処理されるのでこの行でソフトマックスをかける必要はない
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
X_test = X_FromcsvTotensor('X_test')
Y_test = Y_FromcsvTotensor('Y_test')

#損失関数の定義
loss_entropy = nn.CrossEntropyLoss()

#最適化手法の決定
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

EPOCHS = 2000  # 上と同じことを2000回繰り返す
for epoch in range(EPOCHS):
    optimizer.zero_grad()  # 重みとバイアスの更新で内部的に使用するデータをリセット
    outputs = model(X_train)  # 手順1：ニューラルネットワークにデータを入力
    loss = loss_entropy(outputs, Y_train)  # 手順2：正解ラベルとの比較
    loss.backward()  # 手順3-1：誤差逆伝播
    optimizer.step()  # 手順3-2：重みとバイアスの更新
    
    '''if epoch%100==99:
        print(f'損失：{loss}')
        print(f'重み：{model.layer1.weight}')'''

#活性化関数の定義
activation = nn.Softmax(dim=0)

#訓練データでの正解率計算
number = 0
for i in range(len(X_train)):
    if torch.argmax(activation(model(X_train[i]))) == Y_train[i]:
        number += 1
print(f'訓練データにおける正解率：{number/len(X_train)}')

#評価データでの正解率計算
number = 0
for i in range(len(X_test)):
    if torch.argmax(activation(model(X_test[i]))) == Y_test[i]:
        number += 1
print(f'評価データにおける正解率：{number/len(X_test)}')