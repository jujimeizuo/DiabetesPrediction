import pandas as pd
import time
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('./dataset/train.csv')
predict_data = pd.read_csv('./dataset/test_withoutLable.csv')

diabetes_train_data_X = train_data.drop(['ID', 'Diabetes_binary', 'Sex', 'NoDocbcCost', 'AnyHealthcare', 'HvyAlcoholConsump', 'Veggies', 'Fruits', 'CholCheck'], axis=1)
diabetes_train_data_Y = train_data['Diabetes_binary']
diabetes_predict_data = predict_data.drop(['ID', 'Sex', 'NoDocbcCost', 'AnyHealthcare', 'HvyAlcoholConsump', 'Veggies', 'Fruits', 'CholCheck'], axis=1)

train_data, test_data, train_label, test_label = train_test_split(diabetes_train_data_X, diabetes_train_data_Y, test_size=0.2)

train_data = torch.from_numpy(train_data.values).float()
train_label = torch.from_numpy(train_label.values).float()
test_data = torch.from_numpy(test_data.values).float()
test_label = torch.from_numpy(test_label.values).float()
predict_data = torch.from_numpy(diabetes_predict_data.values).float()

train_dataset = TensorDataset(train_data, train_label)
test_dataset = TensorDataset(test_data, test_label)

trainLoader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0)
testLoader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=0)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(14, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.Dropout(p=0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(p=0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.fc(x)

net = Net()

criterion = torch.nn.CrossEntropyLoss() # 损失函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.01) # 优化器

start = time.time()
net.train()
for epoch in range(50):
    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        # 获取数据
        inputs, labels = data
        # 梯度清零
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        # 反向传播
        loss.backward()
        # 更新梯度
        optimizer.step()
        
        running_loss += loss.item()
        if i % 20 == 19:
            print('[%d, %5d] loss: %.5f' % (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

print('Finished Training, Total cost time: ', time.time()-start)

correct = 0
total = 0

with torch.no_grad():
    for data in testLoader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()

print('Accuracy of the network on the 8000 train images: %.5f %%' % (100 * correct / total))

net.eval()
with torch.no_grad():
        outputs = net(predict_data)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.numpy()

submission = pd.read_csv('./dataset/test_withoutLable.csv')
submission = submission[['ID']]
submission['Diabetes_binary'] = predicted
submission.to_csv('./dataset/submission.csv', index=False)
print('ok')