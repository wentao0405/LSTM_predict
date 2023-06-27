import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
# 设置随机种子
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
# 读取数据
df = pd.read_excel('./data/1996-2020年长江寸滩城陵矶流量水位过程20221012.xlsx')  # 假设数据保存在 data.csv 文件中
date = df['时间'].values
df = df.drop(['时间','序号'], axis=1)
X = df.values
y = df['三峡Q2'].values

# 数据归一化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# 划分训练集和测试集
train_size = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

# 自定义数据集类
class MFTDataset(Dataset):
    def __init__(self, data,label, sequence_length):
        self.sequence_length = sequence_length
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label=label

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        MSK=torch.ones(self.sequence_length,self.data.shape[1])
        MSK[self.sequence_length-1,2]=0
        input_seq = self.data[idx:idx + self.sequence_length]*MSK
        target = self.label[idx + self.sequence_length-1]
        return input_seq, target

# LSTM 模型类
class MultiFactorAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(MultiFactorAttention, self).__init__()
        self.attention = nn.MultiheadAttention(input_size, num_heads)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear3=nn.Linear(6,1)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)
        output = self.linear1(attn_output)
        output = torch.relu(output)
        output = self.linear2(output)
        output=output.permute(0,2,1)
        output = self.linear3(output)
        return output.squeeze(1)


# 模型参数
input_size = X.shape[1]
hidden_size = 128
num_layers = 2
output_size = 1
num_epochs = 100
sequence_length=6
# 创建数据集和数据加载器
train_dataset = MFTDataset(X_train, y_train,sequence_length)
test_dataset = MFTDataset(X_test, y_test,sequence_length)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 创建模型和优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiFactorAttention(input_size, hidden_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 模型训练
model.train()
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs = inputs.to(device).to(torch.float32)
        targets = targets.to(device).to(torch.float32)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)*1000000
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
torch.save(model.state_dict(), 'weights/Time_Att.pth')
# 模型评估
model.eval()
predictions = []
real=[]
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device).to(torch.float32)
        targets = targets.to(device).to(torch.float32)
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())
        real.append(targets.cpu().numpy())
predictions = np.concatenate(predictions)
predictions = scaler.inverse_transform(predictions).flatten()  # 将归一化的预测结果逆转回原始范围
real=np.concatenate(real)
real=scaler.inverse_transform(real).flatten()
# 打印预测结果
print(predictions)
plt.plot(np.arange(len(real)),real,'r')
plt.plot(np.arange(len(predictions)),predictions,'b')
plt.show()
dvalue=real-predictions
plt.plot(np.arange(len(real)),dvalue,'y')
plt.show()
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 示例用法

mse = mean_squared_error(real, predictions)
rmse = root_mean_squared_error(real, predictions)
mae = mean_absolute_error(real, predictions)
mape = mean_absolute_percentage_error(real, predictions)

print("均方误差 (MSE):", mse)
print("均方根误差 (RMSE):", rmse)
print("平均绝对误差 (MAE):", mae)
print("相对平均误差 (MAPE):", mape)


