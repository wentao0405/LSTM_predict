import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model import CustomDataset,LSTMModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_excel('./data/1996-2020年长江寸滩城陵矶流量水位过程20221012.xlsx')
date=df['时间'].values
df=df.drop(['序号','时间'],axis=1)
X = df.drop('三峡Q2', axis=1).values
y = df['三峡Q2'].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
#y=scaler.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

input_size = X.shape[1]
hidden_size = 128
num_layers = 3
output_size = 1
batch_size = 32
num_epochs = 10000

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
#model.load_state_dict(torch.load('weights/lstm.pth'))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        inputs=inputs.unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs=outputs.squeeze(1)
        #print(outputs)
        #print(targets)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
torch.save(model.state_dict(), 'weights/lstm.pth')


model.eval()
with torch.no_grad():
    test_loss = 0
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        inputs = inputs.unsqueeze(1)
        targets = targets.to(device)
        outputs = model(inputs)
        outputs = outputs.squeeze(1)
        test_loss += criterion(outputs, targets).item()

    print(f'Test Loss: {test_loss / len(test_loader)}')



