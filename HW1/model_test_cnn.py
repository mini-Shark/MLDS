from  __future__ import absolute_import, division, print_function
import torch.utils.data as torch_data
import torch 
import torch.nn as nn
from cifar10 import CustomCifar10

csv_train = "D:\\test_data\\cifar-10\\train\\data_pair.csv"
csv_test = "D:\\test_data\\cifar-10\\test\\data_pair.csv"
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001
num_epochs = 200
batch_size = 128

train_dataset = CustomCifar10(csv_train) 
train_loader = torch_data.DataLoader(dataset = train_dataset,
                                     batch_size=batch_size,
                                     shuffle=True)# CxHxW

class deep_cnn(nn.Module):
  def __init__(self):
    super(deep_cnn, self).__init__()
    # pytorch's data format is NxCxHxW 
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    # padding=(kernel_size-1)/2 for SAME size after conv
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
    self.fc = nn.Linear(in_features=32*4*4, out_features=10)
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.relu(self.conv1(x))
    out = self.relu(self.conv2(out))
    out = self.maxpool(out)
    out = self.relu(self.conv3(out))
    out = self.relu(self.conv4(out))
    out = self.maxpool(out)
    out = out.reshape(out.size(0), -1)
    out = self.fc(out)

    return out 


model = deep_cnn().to(device)
criterion = nn.CrossEntropyLoss()
# this class combine nn.LogSoftmax and nn.NLLLoss() (Negative Log Likelyhood Loss),
# been used at training classification problem with C class 
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
total_step = len(train_loader)
for epoch in range(num_epochs):
  for i, (x,y) in enumerate(train_loader):
    # forward pass 
    x = x.to(device)
    y = y.to(device)
    outputs = model(x)
    loss = criterion(outputs,y)

    # backward pass and optimization 
    optimizer.zero_grad() # Clears the gradients of all optimized
    loss.backward() # Computes the gradient of current tensor
    optimizer.step() # Performs a single optimization step

    if(i+1)%100==0:
      print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
          .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    