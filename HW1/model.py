import torch
import torch.nn as nn

class deep_model(nn.Module):
    def __init__(self, input_size, output_size):
        super(deep_model, self).__init__()
        self.fc1 = nn.Linear(input_size, 5) 
        self.fc2 = nn.Linear(5, 10,)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)  
        self.fc6 = nn.Linear(10, 10)
        self.fc7 = nn.Linear(10, 5)
        self.fc8 = nn.Linear(5, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.fc7(out)
        out = self.relu(out)
        out = self.fc8(out)
        
        return out

class mid_model(nn.Module):

  def __init__(self,input_size, output_size):
    super(mid_model,self).__init__()
    self.fc1 = nn.Linear(input_size,10)
    self.fc2 = nn.Linear(10,18)
    self.fc3 = nn.Linear(18,15)
    self.fc4 = nn.Linear(15,4)
    self.fc5 = nn.Linear(4,output_size)
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    out = self.relu(out)
    out = self.fc3(out)
    out = self.relu(out)
    out = self.fc4(out)
    out = self.relu(out)
    out = self.fc5(out)

    return out

class shallow_model(nn.Module):

  def __init__(self,input_size, output_size):
    
    super(shallow_model,self).__init__()
    self.fc1 = nn.Linear(input_size,190)
    self.fc2 = nn.Linear(190,output_size)
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)

    return out

def get_model(model_type):

  if model_type=="deep":
    return deep_model
  
  if model_type=="mid":
    return mid_model

  if model_type=="shallow":
    return shallow_model


if __name__=='main':
  pass
    