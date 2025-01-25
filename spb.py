
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm  # make a progress bar
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import math
import matplotlib.ticker as ticker
from torch.optim import lr_scheduler
from IPython.display import display, HTML
import time
import pickle

import datetime

print(f"Torch version: {torch.__version__}")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


def create_sequences(data, window_size, prediction_steps, features, label):
    X = []
    y = []
    for i in range(len(data) - window_size - prediction_steps + 1):
        sequence = data.iloc[i:i + window_size][features]
        target = data.iloc[i + window_size + prediction_steps - 1][label]
        X.append(sequence)
        y.append(target)
    return np.array(X), np.array(y)

def model_metrics_to_dataframe(data_dict):
    """
    Extracts the last values from lists within a dictionary of model statistics
    and creates a DataFrame for each model's modified statistics.

    Args:
    - data_dict (dict): A dictionary where keys represent model names and values
                        contain statistics (possibly including lists)

    Returns:
    - pandas.DataFrame: DataFrame containing modified statistics for each model,
                        concatenated together with keys preserved
    """
    all_dfs = []

    for model_name, model_stats in data_dict.items():
        inner_dict = model_stats.copy()  # Create a copy to avoid modifying the original data

        # Loop through keys and update values if they are lists
        for key, value in inner_dict.items():
            if isinstance(value, list):
                inner_dict[key] = value[-1] if value else None  # Extract last value or None if empty list

        all_dfs.append(pd.DataFrame([inner_dict]))

    return pd.concat(all_dfs)  # Concatenate all DataFrames with keys

class PricePredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(PricePredictionLSTM, self).__init__()
        self.hidden_size = hidden_size  # Size of the hidden state in the LSTM
        self.num_layers = num_layers    # Number of LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # LSTM layer
        self.fc = nn.Linear(hidden_size, output_size)  # Fully connected layer for output prediction

    def forward(self, input_data):
        # Initialize hidden and cell states for LSTM
        initial_hidden = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).to(input_data.device)
        initial_cell = torch.zeros(self.num_layers, input_data.size(0), self.hidden_size).to(input_data.device)

        # Forward propagate through LSTM
        lstm_output, _ = self.lstm(input_data, (initial_hidden, initial_cell))  # Output shape: (batch_size, seq_length, hidden_size)

        # Pass the output of the last time step through the fully connected layer
        last_time_step_output = lstm_output[:, -1, :]  # Extract the output from the last time step
        output = self.fc(last_time_step_output)  # Output shape: (batch_size, output_size)
        return output

class ImprovedPricePredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2):
        super(ImprovedPricePredictionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Multiply by 2 for bidirectional

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size * 2)

        # Apply dropout to the output of the last time step
        out = self.dropout(out[:, -1, :])  # Output shape: (batch_size, hidden_size * 2)

        # Pass the output through the fully connected layer
        out = self.fc(out)  # Output shape: (batch_size, output_size)
        return out

class PricePredictionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(PricePredictionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate GRU
        out, _ = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Pass the output of the last time step through the fully connected layer
        out = self.fc(out[:, -1, :])  # Output shape: (batch_size, output_size)
        return out

class ImprovedPricePredictionGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout=0.2):
        super(ImprovedPricePredictionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # Multiply by 2 for bidirectional
        self.fc = nn.Linear(hidden_size * 2, output_size)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        # Multiply by 2 for bidirectional

        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        # out: tensor of shape (batch_size, seq_length, hidden_size * 2)

        # Apply dropout to the output of the last time step
        out = self.dropout(out[:, -1, :])
        # Output shape: (batch_size, hidden_size * 2)

        # Pass the output through the fully connected layer
        out = self.fc(out)
        # Output shape: (batch_size, output_size)
        return out




adaeur_csv_file = "/home/za/data/kraken.ohlcvt.q1.2024/ADAEUR_60.csv"

#def dateparse (time_in_secs):
#    return datetime.datetime.fromtimestamp(float(time_in_secs))

df = pd.read_csv(adaeur_csv_file,
                 delimiter=',',
                #  parse_dates=True, date_parser=dateparse,
                #  index_col='timestamp',
                 names=["timestamp", "open", "high", "low", "close", "volume", "trades"],
                 header=None)

df['date'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
# df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df


##%matplotlib inline
#
## Downsample the data to not crash the plotting mechanism, we don't need to plot everything in the dataset
#downsampled_df = df.resample('1D').mean()
#
## close price to the left y axis
#plt.plot(downsampled_df.index, downsampled_df['close'], label='Close', color='blue')
#plt.ylabel('Close', color='blue')
#plt.tick_params(axis='y', labelcolor='blue')
#
#plt.xticks(rotation=90, ha='right')
#
## duplicate to get a second y axis on the right and plot the volune
#ax2 = plt.twinx()
#ax2.plot(downsampled_df.index, downsampled_df['volume'], label='Volume', color='red')
#ax2.set_ylabel('Volume', color='red')
#ax2.tick_params(axis='y', labelcolor='red')
#
##plt.xticks(rotation=90)
##plt.tick_params(axis='x', labelrotation=90)
##ax2.tick_params(axis='x', labelrotation=90)
##plt.xticks(rotation=45, ha='right')
#
## Title and legend
#plt.title('Close Price vs. Volume')
#plt.show()
#



all_model_stats = {}

hidden_units = 64 # the amount of internal memory cells of our model, imagine they are small little algorithsm, helping the model to learn
num_layers = 4 # the amount of layers in the model, where each layer contains its own memory cells
learning_rate = 0.001 # the amount the model adapts it's weights and biases (parameters) after every step
learning_rate_step_size=5 # after how many steps should the learning rate be de- or increased?
learning_rate_gamma=0.9 # that's the multiplier to manipulate the learning rate
num_epochs = 300 # how many times (steps) our main loop will go through the training process?
batch_size = 32 # how many data will we process at once?
window_size = 14  # how many data points in the past to look at for our prediction
prediction_steps = 7 # how many data points to skip until the data point that we want to predict
dropout_rate = 0.2 # how many nodes in the model to set to zero

sample_size = 1000

features = ['close', 'volume', 'trades'] # what columns to use
target = 'close' # what column to predict

# right now start with a small sample, as soon as we have enough computing power, we can skip this step
df_sampled = df[features].head(sample_size).copy()

#scaler = MinMaxScaler() # MinMax would work, too, but in fact a stock price has not really "min/max values", except the 0 ;)
scaler = StandardScaler()

# Extract the selected features and transform them
selected_features = df_sampled[features].values.reshape(-1, len(features))
scaled_features = scaler.fit_transform(selected_features)

# Replace the original features with the scaled features in the DataFrame
df_sampled[features] = scaled_features

# MERGING
X, y = create_sequences(df_sampled, window_size, prediction_steps, features, target)

# SPLITTING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# BATCHING
X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train)
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Shuffling is set to "True", if you want to reproduce results, it may help to set shuffle to False
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


###
# Validations
###


# Inverse transform the first set of features and label from the train dataset
first_train_batch = next(iter(train_dataloader))
X_train_batch, y_train_batch = first_train_batch

# Convert PyTorch tensors to NumPy arrays
X_train_np = X_train_batch.numpy()
y_train_np = y_train_batch.numpy()

# Inverse transform features
denormalized_features = scaler.inverse_transform(X_train_np[0])

# Assuming 'features' contains the list of feature column names
features_df = pd.DataFrame(denormalized_features, columns=features)

# Inverse transform the label
label_idx = list(df_sampled.columns).index('close')  # Get the index of the label column
reshaped_label = y_train_np[0]  # No indexing needed for a scalar label
denormalized_label = scaler.inverse_transform([[reshaped_label, 0, 0]])[0][0]

# Create a DataFrame for the label
label_df = pd.DataFrame({'close': [denormalized_label]})

result_df = pd.concat([features_df, label_df], axis=1)

# Format columns (assuming 'decimal_col' contains columns needing decimal formatting and 'thousand_col' contains columns needing thousand formatting)
#result_df['close'] = result_df['close'].applymap(lambda x: "{:.8f}".format(x))
result_df['close'] = result_df['close'].map(lambda x: "{:.8f}".format(x))
result_df['volume'] = result_df['volume'].apply(lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) else x)
result_df['trades'] = result_df['trades'].apply(lambda x: "{:,.0f}".format(x) if isinstance(x, (int, float)) else x)

result_df


### 
# Model
###

# Initialize the model, loss function, and optimizer
#model_1 = PricePredictionLSTM(input_size=len(features), hidden_size=hidden_units, num_layers=num_layers)
#model_1 = ImprovedPricePredictionLSTM(input_size=len(features), hidden_size=hidden_units, num_layers=num_layers, dropout = dropout_rate)
#model_1 = PricePredictionGRU(input_size=len(features), hidden_size=hidden_units, num_layers=num_layers)

model_1 = ImprovedPricePredictionGRU(input_size=len(features), 
                                     hidden_size=hidden_units, 
                                     num_layers=num_layers, 
                                     dropout = dropout_rate)

loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate)
#optimizer = torch.optim.AdamW(model_1.parameters(), lr=learning_rate)
#optimizer = torch.optim.RMSprop(model_1.parameters(), lr=learning_rate)

# Adjust step_size and gamma as needed
scheduler = lr_scheduler.StepLR(optimizer, step_size=learning_rate_step_size, gamma=learning_rate_gamma)



###
# some checking?
###

current_model_stats = {
        'name': model_1.__class__.__name__,
        'device': device,
        'optimizer': optimizer.__class__.__name__,
        'sample_size': sample_size,
        'hidden_units': hidden_units,
        'num_layers': num_layers,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'window_size': window_size,
        'prediction_steps': prediction_steps,
        'dropout_rate': dropout_rate,
        'duration': 0,
        'epochs': [],
        'train_loss_values': [],
        'test_loss_values': [],
        'train_rmse_values': [],
        'test_rmse_values': []
        }

current_model_id = current_model_stats['name'] \
        , current_model_stats['device'] \
        , current_model_stats['optimizer']  \
        , current_model_stats['sample_size']  \
        , current_model_stats['hidden_units']  \
        , current_model_stats['num_layers']  \
        , current_model_stats['learning_rate']  \
        , current_model_stats['batch_size']  \
        , current_model_stats['window_size']  \
        , current_model_stats['prediction_steps']  \
        , current_model_stats['dropout_rate']

current_model_id = '|'.join(map(str, current_model_id))

if current_model_id in all_model_stats:
        last_record = all_model_stats[current_model_id]

        current_model_stats = {
                'name': model_1.__class__.__name__,
                'device': device,
                'optimizer': optimizer.__class__.__name__,
                'sample_size': sample_size,
                'hidden_units': hidden_units,
                'num_layers': num_layers,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'window_size': window_size,
                'prediction_steps': prediction_steps,
                'dropout_rate': dropout_rate,
                'duration': last_record['duration'],
                'epochs': last_record['epochs'],
                'train_loss_values': last_record['train_loss_values'],
                'test_loss_values': last_record['test_loss_values'],
                'train_rmse_values': last_record['train_rmse_values'],
                'test_rmse_values': last_record['test_rmse_values']
        }

current_model_id



###
# 31 loop / running
###


start = time.time()

epoch_count = current_model_stats['epochs']
start_epoch = 0 if len(epoch_count) == 0 else epoch_count[-1]
# helpful if you start over this particular cell

for epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
    # tqdm is our progress bar wrapper

    model_1.train() # activate training mode

    # handle loss monitoring
    total_train_loss = 0.0
    all_train_targets = []
    all_train_outputs = []

    # process batches in the training dataloader
    for batch_idx, (inputs, targets) in enumerate(train_dataloader):

        outputs = model_1(inputs) # calculate predictions
        loss = loss_fn(outputs.squeeze(), targets) # calculat the loss
        optimizer.zero_grad() # reset gradients
        loss.backward() # backward propagation
        optimizer.step() # update parameters

        total_train_loss += loss.item()

        all_train_targets.extend(targets.numpy())
        all_train_outputs.extend(outputs.detach().numpy())

    # scheduler.step()

    model_1.eval() # activate eval mode

    # handle loss monitoring
    total_test_loss = 0.0
    all_test_targets = []
    all_test_outputs = []

    # process batches in the testing dataloader
    for i, (inputs, targets) in enumerate(test_dataloader):
        with torch.inference_mode(): # activate inference mode/no grad
            outputs = model_1(inputs) # calculate predictions
            loss = loss_fn(outputs.squeeze(), targets) # calculate loss

            # monitor loss
            total_test_loss += loss.item()
            all_test_targets.extend(targets.numpy())
            all_test_outputs.extend(outputs.detach().numpy())

    # calculate average epoch losses
    average_epoch_train_loss = total_train_loss / len(train_dataloader)
    average_epoch_test_loss = total_test_loss / len(test_dataloader)

    # caculate accuracy
    train_rmse = math.sqrt(mean_squared_error(all_train_targets, all_train_outputs))
    test_rmse = math.sqrt(mean_squared_error(all_test_targets, all_test_outputs))

    # VISUALIZE
    current_model_stats['epochs'].append(epoch)
    current_model_stats['train_loss_values'].append(average_epoch_train_loss)
    current_model_stats['test_loss_values'].append(average_epoch_test_loss)
    current_model_stats['train_rmse_values'].append(train_rmse)
    current_model_stats['test_rmse_values'].append(test_rmse)

    # LOG
    if epoch % int(num_epochs / 10) == 0 or epoch == num_epochs - 1:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch + 1}/{start_epoch + num_epochs}], "
        f"Train Loss: {average_epoch_train_loss:.4f} | "
        f"Test Loss: {average_epoch_test_loss:.4f} | "
        f"Train RMSE: {train_rmse:.4f} | "
        f"Test RMSE: {test_rmse:.4f} | "
        f"Current LR: {current_lr:.8f} | "
        f"Duration: {time.time() - start:.0f} seconds")

current_model_stats['duration'] += time.time() - start

all_model_stats[current_model_id] = current_model_stats



###
# 32 After Running, Validation & Visualization
###


validation_size = 10000
# df_validation = df[features].head(window_size + prediction_steps).copy() just test one window
df_validation = df[features].head(validation_size).copy()

# actuall we don't need to re-initialize the scaler again
# scaler = MinMaxScaler()
# scaler = StandardScaler()

# Extract the selected features and transform them
selected_features = df_validation[features].values.reshape(-1, len(features))
scaled_features = scaler.fit_transform(selected_features)

# Replace the original features with the scaled features in the DataFrame
df_validation[features] = scaled_features

# MERGING
X_validate, y_validate = create_sequences(df_validation, window_size, prediction_steps, features, target)

# BATCHING
# Convert NumPy arrays to PyTorch tensors
X_validate_tensor = torch.Tensor(X_validate)
y_validate_tensor = torch.Tensor(y_validate)

model_1.eval()
# Iterate over the test DataLoader to generate predictions
with torch.inference_mode():
    output = model_1(X_validate_tensor)

predicted_array = np.array(output).reshape(-1, 1)
dummy_columns = np.zeros((predicted_array.shape[0], 2))  # Assuming 2 dummy columns
predicted_array_with_dummy = np.concatenate((predicted_array, dummy_columns), axis=1)
predicted_close_with_dummy = scaler.inverse_transform(predicted_array_with_dummy)
predicted_close = predicted_close_with_dummy[:, :-2]  # Remove the last two columns
predicted_close



###
# 33 plot
###


plt.figure(figsize=(12, 5))

# PLOT LOSS
plt.subplot(1, 3, 1)
plt.plot(epoch_count, current_model_stats['train_loss_values'], label='Train Loss')
plt.plot(epoch_count, current_model_stats['test_loss_values'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training and Test Loss after {current_model_stats["duration"]:.0f} seconds', fontsize=10)
plt.legend()

# PLOT RMSE
plt.subplot(1, 3, 2)
plt.plot(epoch_count, current_model_stats['train_rmse_values'], label='Train RMSE', color='blue')
plt.plot(epoch_count, current_model_stats['test_rmse_values'], label='Test RMSE', color='red')

plt.ylabel('Accuracy (RMSE)')
plt.xlabel('Epochs')
plt.title(f'Training and Test Accuracy (RMSE) after {current_model_stats["duration"]:.0f} seconds', fontsize=10)
plt.legend(loc='upper right')

# PLOT PREDICTIONS
df_visualize = df.head(validation_size).copy()
y_axis_max = max(np.amax(predicted_close), max(df_visualize['close'])) * 1.25
y_axis_min = min(np.amin(predicted_close), min(df_visualize['close'])) * 0.75

plt.subplot(1, 3, 3)

plt.plot(df_visualize.index, df_visualize['close'], label='Actual', color='blue')
plt.ylabel('Close Price')
plt.legend()
plt.legend(loc='upper left')
plt.ylim(y_axis_min, y_axis_max)

plt.twinx()  # Create a second y-axis sharing the same x-axis

nan_values = np.full(window_size + prediction_steps - 1, np.nan)
predicted_close_with_nan = np.concatenate([nan_values, predicted_close.ravel()])

plt.plot(df_visualize.index, predicted_close_with_nan, label='Predicted', color='red')
plt.ylim(y_axis_min, y_axis_max)

plt.xlabel('Time')
plt.title('Actual vs. Predicted Data', fontsize=10)

# Set different y-axis for actual and predicted data
ax2.set_ylabel('Predicted Data')

# Adjust x-axis ticks and labels for better readability
plt.xticks(fontsize=8)  # Set font size for x-axis ticks
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))  # Set the number of ticks (change nbins as needed)
plt.legend()
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

