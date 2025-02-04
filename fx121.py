import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

import numpy as np


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)


#model = nn.Sequential(
#    Lambda(preprocess),
#    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
#    nn.ReLU(),
#    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
#    nn.ReLU(),
#    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
#    nn.ReLU(),
#    nn.AvgPool2d(4),
#    Lambda(lambda x: x.view(x.size(0), -1)),
#)


# 1, 10, 100, 10, 1
model = nn.Sequential(
    nn.Linear(1, 10),
    nn.Tanh(),
    nn.Linear(10, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.Tanh(),
    nn.Linear(10, 1)
)

# sigmoid() is the logistic function, 
# and that is a rescaled and shifted version of tanh(). 
# Given that the weights in Linear layers do scaling and their biases do shifts, 
# ... where sigmoid() and tanh() act essentially equivalently.

lr = 1E-3
opt = optim.SGD(model.parameters(), lr=lr)

#fit(epochs, model, loss_func, opt, train_dl, valid_dl)


# data 

x = np.linspace(-20.0, 20.0, 100, dtype=np.float32)
y = np.sin(x/3.14)
# print(x.shape)
# print(y.shape)

# shape the data
xs = x.reshape((x.shape[0], 1))
ys = y.reshape((y.shape[0], 1))

ttx = torch.tensor(xs, dtype=torch.float32)
tty = torch.tensor(ys, dtype=torch.float32)

dsxy = TensorDataset(ttx, tty)
dlxy = DataLoader(dsxy, batch_size=32, shuffle=True)

#iter(dlxy)


import torch.nn.functional as F

loss_func = F.cross_entropy


epochs = 100
model.train()
for epoch in range(epochs):

    pred = model(ttx)
    loss = loss_func(pred.squeeze(), tty.squeeze())
    if (epoch + 1) % 5 == 0: 
        print(f"epoch {epoch}, loss {loss.item()}")

    with torch.no_grad():
        loss.backward()
        opt.step()
        opt.zero_grad()


import matplotlib.pyplot as plt
plt.scatter(ttx.squeeze().numpy(), pred.detach().squeeze().numpy())
plt.show()





