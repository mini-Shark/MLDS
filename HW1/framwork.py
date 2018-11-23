from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from model import *

class Regressor():

  def __init__(self, num_epochs, batch_size, learning_rate, device):

    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.device = device
    self.model = None
    self.loss_epoch = [] 

  def train(self, x_train, y_train): 
    '''
    x_train: tensor, training feature
    y_train: tensor, training labels
    note: objected python program will automaticlly add "self" parameters when call object's method 
    So there should add "self" parameters when we define method 
    '''
    assert x_train.shape[0] == x_train.shape[0], "x, y should have same size at dimension 0"
    assert self.model != None, "model is none"

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=self.batch_size,
                                               shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    total_step = len(train_loader)

    for epoch in range(self.num_epochs):
        loss_batch = []
        for i, (x, y) in enumerate(train_loader):  
            # Move tensors to the configured device
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass
            outputs = self.model(x)
            loss = criterion(outputs, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.item())
            if (i+1) % 8 == 0 and (epoch+1)%10==0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}' 
                        .format(epoch+1, self.num_epochs, i+1, total_step, loss.item()))

        self.loss_epoch.append(np.array(loss_batch).mean())  

  def predict(self, x_test):
    '''
    x_test: tensor, testing feature
    '''
    test_dataset = torch.utils.data.TensorDataset(x_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=self.batch_size, 
                                              shuffle=True)
    with torch.no_grad():
      X = []
      P = []
      for x in test_loader:
          x = x[0].to(self.device)
          outputs = self.model(x)
          X.append(x.numpy())
          P.append(outputs.numpy())

      x = np.squeeze(np.concatenate(X))
      pred = np.squeeze(np.concatenate(P))
      index = np.argsort(x)

    return pred[index]

