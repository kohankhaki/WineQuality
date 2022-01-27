# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import torchviz
from torchsummary import summary
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        D_in = 11
        H = 64
        D_out = 1
        self.l1 = torch.nn.Linear(D_in, H)
        self.lm = torch.nn.Linear(H, D_out)
        self.lv = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        m = self.lm(x)
        v = self.lv(x)
        v = F.softplus(v) + 10 ** -6
        return m, v
        # return m, np.zeros(m.shape)


def normalize(x):
	return x/np.linalg.norm(x)

winequality = pd.read_csv('winequality-red.csv', ';').values
winequality.astype(np.float32)
winequality = np.array(list(map(normalize, winequality.transpose()))).transpose()

train, validate, test = np.split(winequality, [int(.6*len(winequality)), int(.8*len(winequality))])

train_wine = torch.from_numpy(train[:,:-1]).type('torch.FloatTensor')
train_quality = torch.from_numpy(train[:,-1:]).type('torch.FloatTensor')

validate_wine = torch.from_numpy(validate[:,:-1]).type('torch.FloatTensor')
validate_quality = torch.from_numpy(validate[:,-1:]).type('torch.FloatTensor')

test_wine = torch.from_numpy(test[:,:-1]).type('torch.FloatTensor')
test_quality = torch.from_numpy(test[:,-1:]).type('torch.FloatTensor')

best_loss = 1
best_model = None
best_hidden = None

# loss_fn = torch.nn.MSELoss(reduction='sum')

def loss_fn(mean, sigma, train_quality):
	mse = (mean - train_quality) ** 2
	x = mse / ((2 * sigma)) + 0.5 * torch.log(sigma)
	# x = mse
	loss = torch.mean(x)
	return loss

for hidden in range(3, 4, 1):
	N, D_in, H, D_out = int(.6*len(winequality)), 11, hidden, 1

	# model = torch.nn.Sequential(
	#     torch.nn.Linear(D_in, H),
	#     torch.nn.ReLU(),
	#     torch.nn.Linear(H, D_out),
	# )
	
	# model = torch.nn.Sequential(
	#     torch.nn.Linear(D_in, H),
	#     torch.nn.ReLU(),
	#     torch.nn.Linear(H, D_out * 2),
	# )
	model = Net().to(torch.device("cpu"))
    # model = Net().to(torch.device("cuda" if use_cuda else "cpu"))
	prev_loss = 0
	optimizer = optim.Adadelta(model.parameters(), lr=1e-3)
	# learning_rate = 1e-3
	for t in range(50000):
		m, v = model(train_wine)
		loss = loss_fn(m, v, train_quality)
		print(t, loss.item())

		if abs(prev_loss - loss.item()) < 1e-8:
			break

		prev_loss = loss.item()

		optimizer.zero_grad()

		loss.backward()
		optimizer.step()


		# with torch.no_grad():
		# 	for param in model.parameters():
		# 		param -= learning_rate * param.grad


	m, v = model(validate_wine)
	loss = loss_fn(m, v, validate_quality)
	print(hidden, loss.item())

	if loss < best_loss:
	    best_loss = loss
	    best_model = model
	    best_hidden = hidden
	    
m, v = best_model(test_wine)
loss = loss_fn(m, v, test_quality)
print(loss.item())

print(m.shape, test_quality[:, 0].shape)
x = (m[:, 0].detach() - test_quality[:, 0]) ** 2
print(torch.mean(x))
y = v[:, 0].detach()
print(x.shape, y.shape)
plt.plot(x.detach().numpy(), y.detach().numpy(), 'ro')
plt.show()




# dot = torchviz.make_dot(pred_quality)
# dot.format = 'jpeg'
# dot.render('/Users/farnazkohankhaki/Projects/WineQuality/winequality-model')
# summary(best_model, input_size=(1, best_hidden, 11))

