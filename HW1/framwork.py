from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import *

class frame_work():

  def __init__(self, input_size, output_size,
               num_epochs, batch_size, learning_rate, device):

    self.input_size = input_size
    self.output_szie = output_size
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.device = device
    self.model = deep_model(input_size, output_size).to(device)
    self.loss_epoch = [] 

  def train(x_train, y_train):
    '''
    x_train: tensor, training feature
    y_train: tensor, training labels
    '''
    assert x.shape[0] == y.shape[0], "x, y should have same size at dimension 0"
    assert self.model != None, "model is none"

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=self.batch_size,
                                               shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        loss_batch = []
        for i, (x, y) in enumerate(train_loader):  
            # Move tensors to the configured device
            x = x.to(device)
            y = y.to(device)

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
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        self.loss_epoch.append(np.array(loss_batch).mean())  

  def validation(x_test, y_test):
    '''
    x_test: tensor, training feature
    y_test: tensor, training labels
    '''
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=self.batch_size, 
                                              shuffle=True)
    with torch.no_grad():
      X = []
      P = []
      for x, y in test_loader:
          x = x.to(device)
          y = y.to(device)
          outputs = model(x)
          X.append(x.numpy())
          P.append(outputs.numpy())

      x = np.squeeze(np.concatenate(X))
      pred = np.squeeze(np.concatenate(P))
      index = np.argsort(x)

