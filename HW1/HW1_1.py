from __future__ import absolute_import

from model import get_model
import matplotlib.pyplot as plt 
from framwork import Regressor
import torch
import argparse
import numpy as np 


parser = argparse.ArgumentParser(description="deep network used to fit function")
parser.add_argument('--seed', type=int, default=2018, help="random seed")
parser.add_argument('--num_epochs', type=int, default=1000, help="training epochs")
parser.add_argument('--learning_rate','-lr', type=float, default=0.001, help="optimizer learning rate")
parser.add_argument('--batch_size','-b',type=int, default=128, help="batch size of training (should less than 1024)")
parser.add_argument('--model','-m', type=str, default="deep", choices=["deep", "mid", "shallow"], help="model type")
args = parser.parse_args() 

# seed = args.seed
input_size = 1
output_size = 1
# num_epochs = 200
# batch_size = 128
# learning_rate = 0.001
# model_type = "shallow"

torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

regressor = Regressor(args.num_epochs, args.batch_size, args.learning_rate, device)
model = get_model(args.model)
regressor.model = model(input_size, output_size).to(device)

x_train = np.linspace(0.0001,1,1024)
y_train = list(map(lambda x: (np.sin(5*np.pi*x))/(5*np.pi*x),x_train))
x_train = x_train[:,np.newaxis]
y_train = np.array(y_train)[:,np.newaxis]

regressor.train(torch.Tensor(x_train), torch.Tensor(y_train))

x_test = np.linspace(0.00001,1,512) 
y_true = list(map(lambda x: (np.sin(5*np.pi*x))/(5*np.pi*x),x_test))
x_test = x_test[:,np.newaxis]
y_true = np.array(y_true)[:,np.newaxis]

y_pred = regressor.predict(torch.Tensor(x_test))

fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3 ,figsize=[16,8])
ax1.plot(np.squeeze(x_train), np.squeeze(y_train), 'b')
ax1.set_title("origin function curve")
ax2.plot(x_test, y_pred, '.')
ax2.set_title("fited function curve")
ax3.plot(list(range(regressor.num_epochs)),regressor.loss_epoch,'-')
ax3.set_title("loss curve")
plt.show()


