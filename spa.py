import pandas as pd

import datetime


#df = pd.read_csv("/home/za/data/kraken.ohlcvt.q1.2024/ADAEUR_60.csv")
#df = pd.read_csv("/home/za/data/kraken.ohlcvt.q1.2024/ADAEUR_60.csv",
#                 header=None,
#                 names=["date", "open", "high", "low", "close", "volume", "trades"],
#                 dtype={"date": Datetime})

def dateparse (time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))

df = pd.read_csv("/home/za/data/kraken.ohlcvt.q1.2024/ADAEUR_60.csv",
                 delimiter=',', 
                 parse_dates=True, date_parser=dateparse, index_col='date', 
                 names=["date", "open", "high", "low", "close", "volume", "trades"],
                 header=None)


#df['date'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
#df['date'] = pd.to_datetime(df['date'])
#df.set_index('date', inplace=True)
df


# Downsample the data to not crash the plotting mechanism, we don't need to plot everything in the dataset
downsampled_df = df.resample('1D').mean()

# close price to the left y axis
plt.plot(downsampled_df.index, downsampled_df['close'], label='Close', color='blue')
plt.ylabel('Close', color='blue')
plt.tick_params(axis='y', labelcolor='blue')

# duplicate to get a second y axis on the right and plot the volune
ax2 = plt.twinx()
ax2.plot(downsampled_df.index, downsampled_df['volume'], label='Volume', color='red')
ax2.set_ylabel('Volume', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Title and legend
plt.title('Close Price vs. Volume')
plt.show()
