
#!/usr/bin/env python
# coding: utf-8



# For tips on running notebooks in Google Colab, see
# https://pytorch.org/tutorials/beginner/colab
#get_ipython().run_line_magic('matplotlib', 'inline')


# 
# Quickstart
# ==========
# 
# This section runs through the API for common tasks in machine learning.
# Refer to the links in each section to dive deeper.
# 
# Working with data
# -----------------
# 
# PyTorch has two [primitives to work with
# data](https://pytorch.org/docs/stable/data.html):
# `torch.utils.data.DataLoader` and `torch.utils.data.Dataset`. `Dataset`
# stores the samples and their corresponding labels, and `DataLoader`
# wraps an iterable around the `Dataset`.
# 


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# PyTorch offers domain-specific libraries such as
# [TorchText](https://pytorch.org/text/stable/index.html),
# [TorchVision](https://pytorch.org/vision/stable/index.html), and
# [TorchAudio](https://pytorch.org/audio/stable/index.html), all of which
# include datasets. For this tutorial, we will be using a TorchVision
# dataset.
# 
# The `torchvision.datasets` module contains `Dataset` objects for many
# real-world vision data like CIFAR, COCO ([full list
# here](https://pytorch.org/vision/stable/datasets.html)). In this
# tutorial, we use the FashionMNIST dataset. Every TorchVision `Dataset`
# includes two arguments: `transform` and `target_transform` to modify the
# samples and labels respectively.
# 



# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break






# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


# Read more about [building neural networks in
# PyTorch](buildmodel_tutorial.html).
# 

# ------------------------------------------------------------------------
# 

# Optimizing the Model Parameters
# ===============================
# 
# To train a model, we need a [loss
# function](https://pytorch.org/docs/stable/nn.html#loss-functions) and an
# [optimizer](https://pytorch.org/docs/stable/optim.html).
# 

# In[ ]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# In a single training loop, the model makes predictions on the training
# dataset (fed to it in batches), and backpropagates the prediction error
# to adjust the model\'s parameters.
# 

# In[ ]:


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# We also check the model\'s performance against the test dataset to
# ensure it is learning.
# 

# In[ ]:


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


train(train_dataloader, model, loss_fn, optimizer)
test(test_dataloader, model, loss_fn)



