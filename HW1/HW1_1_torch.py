# from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from model import *

seed = 2018
input_size = 1
output_size = 1
num_epochs = 200
batch_size = 128
learning_rate = 0.001

torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_train = np.linspace(0.0001,1,1024)
y_train = list(map(lambda x: (np.sin(5*np.pi*x))/(5*np.pi*x),x_train))
x_train = x_train[:,np.newaxis]
y_train = np.array(y_train)[:,np.newaxis]
x_tensor = torch.Tensor(x_train)
y_tensor =  torch.Tensor(y_train)
train_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
x_test = np.linspace(0.00001,1,512) 
y_test = list(map(lambda x: (np.sin(5*np.pi*x))/(5*np.pi*x),x_test))
x_test = x_test[:,np.newaxis]
y_test = np.array(y_test)[:,np.newaxis]
x_tensor = torch.Tensor(x_test)
y_tensor =  torch.Tensor(y_test)
test_dataset = torch.utils.data.TensorDataset(x_tensor)
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)


model = deep_model(input_size, output_size).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

# Train the model
total_step = len(train_loader)
loss_epoch = []
for epoch in range(num_epochs):
    loss_batch = []
    for i, (x, y) in enumerate(train_loader):  
        # Move tensors to the configured device
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_batch.append(loss.item())
        if (i+1) % 8 == 0 and (epoch+1)%10==0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    loss_epoch.append(np.array(loss_batch).mean())

# Test the model
with torch.no_grad():

    X = []
    P = []
    for x in test_loader:
        x = x[0].to(device)
        # y = y.to(device)
        outputs = model(x)
        X.append(x.numpy())
        P.append(outputs.numpy())

x = np.squeeze(np.concatenate(X))
pred = np.squeeze(np.concatenate(P))
index = np.argsort(x)


f, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3 ,figsize=[16,8])
ax1.plot(np.squeeze(x_train), np.squeeze(y_train), 'b')
ax1.set_title("origin function curve")
ax2.plot(x[index], pred[index], '.')
ax2.set_title("fited function curve")
ax3.plot(list(range(num_epochs)),loss_epoch,'-')
ax3.set_title("loss curve")
plt.show()