
import pandas as pd


# ADA is the crypto currency, Eur means price in Euro.
adaeur_csv_file = "/home/za/data/kraken.ohlcvt.q1.2024/ADAEUR_60.csv"

# Kraken name BTC as XBT, it's by standard org.
bitcoin_csv_file = "/home/za/data/kraken.trading.history.q1.2024/XBTUSD.csv"


# read into pandas dataframe
# the .csv file has no header, so names[] is the headers/titles
def read_adaeur_2024_q1():
    df = pd.read_csv(adaeur_csv_file,
                     delimiter=',',
                     names=["timestamp", "open", "high", "low", "close", "volume", "trades"],
                     header=None)
    return df


def read_btcusd_2024_q1():
    df = pd.read_csv(bitcoin_csv_file)
#                     delimiter=',',
#                     names=["timestamp", "open", "high", "low", "close", "volume", "trades"],
#                     header=None)
    return df


