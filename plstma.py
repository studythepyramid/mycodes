
import dfread

import torch
nn = torch.nn

import numpy as np

#from torch.optim import lr_scheduler


# read in Bitcoin price csv file to pandas Dataframe, 
# with price and volume data during in 2024 Q1
btcdf = dfread.read_btcusd_2024_q1()

usd = btcdf.iloc[:,1]
npusd = usd.to_numpy()
#tusd = torch.Tensor(usd.to_numpy())

u100 = usd[:100]


## 
# useless
def split_data(stock, lookback):
    data_raw = stock.to_numpy() # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);

    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]

    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]

    return [x_train, y_train, x_test, y_test]

lookback = 20 # choose sequence length
#x_train, y_train, x_test, y_test = split_data(u100, lookback)


# pl=priceList, ws=windowSize
def mysplit(pl, ws):
    pla = np.array(pl)
    plaChunks = int(len(pla)/ws)

    if( (plaChunks * ws) != len(pla)):
        raise Exception(f"The length of pl {len(pl)} not fit ws {ws}")

    plb = pla.reshape((plaChunks, ws))

    x = np.copy(plb)
    y = np.copy(plb[1:])

    # len y is 1 less than original plb
    seqLen = len(y)
    h80p = int(seqLen * 0.8)

    return x, y

priceList = u100
windowSize = 10
xx, yy = mysplit(u100, windowSize)







# get rid of the extra head, cut to 10 times/length
start = len(npusd) % 10
cutted_npusd = npusd[start:]
# convert to tensor
tprice = torch.Tensor(cutted_npusd)

# p10 is chunks of the price, each has 10 prices.
p10 = tprice.reshape((-1,10))


## another approach to prepare data
from sklearn.preprocessing import MinMaxScaler
price = cutted_npusd  #...

#scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = MinMaxScaler(feature_range=(0, 1))

#price['Close'] = scaler.fit_transform(price['Close'].values.reshape(-1,1))
p2d = price.reshape((-1,1))
pscaled = scaler.fit_transform(p2d)



## get 5 line/rows of data to test
i5 = p10[:5,]
#i5 = i5.reshape((5,1,10))
#ti5 = torch.Tensor(i5)

## LSTM model input:
## (L, N, Hin), 
## sequence Length, batch size, Hin size
## omit N first

## input
## L = 5, Hin = 10
## h_0, c_0
## num_of_layers, Hout/Hcell = Hidden Size?



rnn = nn.LSTM(10, 20, 2)
# 2 layers, with hidden size 20, 
# ommit the batch size = 1
h0 = torch.randn(2, 20)
c0 = torch.randn(2, 20)
#output, (hn, cn) = rnn(ti5, (h0, c0))





## start to run training
#model = nn.LSTM(input_dim=10, hidden_dim=20, output_dim=10, num_layers=2)
model = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
#
criterion = torch.nn.MSELoss(reduction='mean')
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

import time
num_epochs = 100
hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []

#, we train the model over 100 epochs.
#for t in range(num_epochs):

#y_train_pred = model(x_train)
#y_train_pred = model(i5)



#loss = criterion(y_train_pred, y_train_lstm)
#print("Epoch ", t, "MSE: ", loss.item())
#hist[t] = loss.item()
#optimiser.zero_grad()
#loss.backward()
#optimiser.step()
#
#training_time = time.time()-start_time
#print("Training time: {}".format(training_time))


### end1

#learning_rate = 1E-3
#
#
#class PriceLSTM(nn.Module):
#    def __init__(self, input_size, hidden_size, num_layers, output_size):
#        super().__init__()
#        self.hidden_size = hidden_size  # Size of the hidden state in the LSTM
#        self.num_layers = num_layers    # Number of LSTM layers
#        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # LSTM layer
#        self.fc = nn.Linear(hidden_size, output_size)  # Fully connected layer for output prediction
#
#    def forward(self, input_data):
#        # Initialize hidden and cell states for LSTM
#        initial_hidden = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).to(input_data.device)
#        initial_cell = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).to(input_data.device)
#
#        # Forward propagate through LSTM
#        # Output shape: (batch_size, seq_length, hidden_size)
#        lstm_output, _ = self.lstm(input_data, (initial_hidden, initial_cell))
#
#        # Pass the output of the last time step through the fully connected layer
#        # Extract the output from the last time step
#        last_time_step_output = lstm_output[:, -1, :]
#        output = self.fc(last_time_step_output)
#        # Output shape: (batch_size, output_size)
#        return output
#
#
#
#model = PriceLSTM(
#        input_size=2,  # price and volume?
#        hidden_size=64, 
#        num_layers=4,
#        output_size=10)
##, dropout = 0.2)
#
#loss_fn = nn.MSELoss()
#
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
## Adjust step_size and gamma as needed
##scheduler = lr_scheduler.StepLR(optimizer, step_size=learning_rate_step_size, gamma=learning_rate_gamma)
#
#
#### test input and model
#
#model.train() # activate training mode


# outputs = model(inputs) # calculate predictions
# loss = loss_fn(outputs.squeeze(), targets) # calculat the loss
# optimizer.zero_grad() # reset gradients
# loss.backward() # backward propagation
# optimizer.step() # update parameters
    




## In[998]:
#
#
#start = time.time()
#
#epoch_count = current_model_stats['epochs']
#start_epoch = 0 if len(epoch_count) == 0 else epoch_count[-1] # helpful if you start over this particular cell
#
#
#def run():
#    for epoch in range(start_epoch, start_epoch + num_epochs):
#
#        model.train() # activate training mode
#
#        # handle loss monitoring
#        total_train_loss = 0.0
#        all_train_targets = []
#        all_train_outputs = []
#        
#        # process batches in the training dataloader
#        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
#            
#            outputs = model(inputs) # calculate predictions
#            loss = loss_fn(outputs.squeeze(), targets) # calculat the loss
#            optimizer.zero_grad() # reset gradients
#            loss.backward() # backward propagation
#            optimizer.step() # update parameters
#            
#            total_train_loss += loss.item()
#
#            all_train_targets.extend(targets.numpy())
#            all_train_outputs.extend(outputs.detach().numpy())
#
#        # scheduler.step()
#
#        model.eval() # activate eval mode
#
#        # handle loss monitoring
#        total_test_loss = 0.0
#        all_test_targets = []
#        all_test_outputs = []
#
#        # process batches in the testing dataloader
#        for i, (inputs, targets) in enumerate(test_dataloader):
#            with torch.inference_mode(): # activate inference mode/no grad
#                outputs = model(inputs) # calculate predictions
#                loss = loss_fn(outputs.squeeze(), targets) # calculate loss
#
#                # monitor loss
#                total_test_loss += loss.item()
#                all_test_targets.extend(targets.numpy())
#                all_test_outputs.extend(outputs.detach().numpy())
#
#        # calculate average epoch losses
#        average_epoch_train_loss = total_train_loss / len(train_dataloader)
#        average_epoch_test_loss = total_test_loss / len(test_dataloader)
#        
#        # caculate accuracy
#        train_rmse = math.sqrt(mean_squared_error(all_train_targets, all_train_outputs))
#        test_rmse = math.sqrt(mean_squared_error(all_test_targets, all_test_outputs))
#
#        # VISUALIZE
#        current_model_stats['epochs'].append(epoch)
#        current_model_stats['train_loss_values'].append(average_epoch_train_loss)
#        current_model_stats['test_loss_values'].append(average_epoch_test_loss)
#        current_model_stats['train_rmse_values'].append(train_rmse)
#        current_model_stats['test_rmse_values'].append(test_rmse)
#
#        # LOG
#        if epoch % int(num_epochs / 10) == 0 or epoch == num_epochs - 1:
#            current_lr = scheduler.get_last_lr()[0]
#            print(f"Epoch [{epoch + 1}/{start_epoch + num_epochs}], "
#            f"Train Loss: {average_epoch_train_loss:.4f} | "
#            f"Test Loss: {average_epoch_test_loss:.4f} | "
#            f"Train RMSE: {train_rmse:.4f} | "
#            f"Test RMSE: {test_rmse:.4f} | "
#            f"Current LR: {current_lr:.8f} | "
#            f"Duration: {time.time() - start:.0f} seconds")
#
#            pass
#        pass
#    pass
#
#
#current_model_stats['duration'] += time.time() - start
#
#all_model_stats[current_model_id] = current_model_stats
#



