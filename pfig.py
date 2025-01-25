
import yfinance as yf
import plotly.graph_objects as go

# Retrieve AAPL historical data
symbol = "AAPL"
ticker = yf.Ticker(symbol)
data = ticker.history(period="1mo")

# Create candlestick chart
fig = go.Figure(data=[go.Candlestick(x=data.index,
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'])])

# Customize the chart layout
fig.update_layout(title=f"{symbol} Candlestick Chart (1 Month)",
                  yaxis_title="Price",
                  xaxis_rangeslider_visible=False)

# Display the chart
fig.show()
