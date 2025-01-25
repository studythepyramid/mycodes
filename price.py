import torch
from torch import nn

import numpy  as np
import pandas as pd

#from torch.utils.data import Dataset

from torch.utils.data import TensorDataset
from torch.utils.data import StackDataset

from torch.utils.data import DataLoader


# ADA is the crypto currency, Eur means price in Euro.
adaeur_csv_file = "/home/za/data/kraken.ohlcvt.q1.2024/ADAEUR_60.csv"


# Get cpu, gpu or mps device for training.
device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
#print(f"Using {device} device")





# read into pandas dataframe
# the .csv file has no header, so names[] is the headers/titles
df = pd.read_csv(adaeur_csv_file,
                 delimiter=',',
                 names=["timestamp", "open", "high", "low", "close", "volume", "trades"],
                 header=None)

df['date'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
df.set_index('date', inplace=True)



batch_size = 32
batch_size = 1


colNames = ['timestamp', 'close', 'volume']
targetColName = 'close'

closePrice = df[['timestamp', 'close']].to_numpy()

dfLen = len(df)


numCol = 1

lws = left_window_size  = sample_length = 100
rws = right_window_size = prediction_length = 10

inLen = lws * numCol
outLen = rws * numCol


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        #self.flatten = torch.flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(inLen, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, outLen)
        )

    def forward(self, x):
        x = torch.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

#model = NeuralNetwork().to(device)
model = NeuralNetwork()
#print(model)



# make up training data, 
def sliceData(data, lws, rws):
    X = []
    y = []

    dLen = len(data)

    for i in range(len(data) - lws - rws - 1):
        left = data[i :i + lws]
        right = data[i + lws : i + lws + rws]
        X.append(left)
        y.append(right)
    return np.array(X), np.array(y)


X, y = sliceData(closePrice, lws , rws)

X_tensor = torch.Tensor(X)
y_tensor = torch.Tensor(y)

tDataset = TensorDataset(X_tensor, y_tensor)


# pddf is pandas Dataframe, CSV data read by Pandas
# lws rws, left/right window size
def preDataLoader(pddf, lws, rws):
    #closePrice = pddf[['timestamp', 'close']].to_numpy()
    closePrice = pddf[['close']].to_numpy()

    X, y = sliceData(closePrice, lws, rws)

    X_tensor = torch.Tensor(X)
    y_tensor = torch.Tensor(y)

    dset = StackDataset(X_tensor, y_tensor)

    dloader = DataLoader(dset, batch_size=1, shuffle=False)
    return dloader, dset


# for test
dloader, dset = preDataLoader(df, lws, rws)
#b, X, y = 


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)

        yflat = torch.flatten(y)
        loss = loss_fn(pred, yflat)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


#train_dataloader = DataLoader(tDataset, batch_size=1, shuffle=False)

#epochs = 5
epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(dloader, model, loss_fn, optimizer)
    #test(test_dataloader, model, loss_fn)
print("Done!")



#train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
#test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


