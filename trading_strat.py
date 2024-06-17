import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

data = yf.download('TSLA', start='2024-06-01', end='2024-06-4')

data.fillna(method='ffill', inplace=True)

#Trading Strategy
def moving_average_crossover(data, short_window=10, long_window=30):
    data['SMA_short'] = data['Close'].rolling(window=short_window).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_window).mean()
    
    data['Signal'] = 0
    data.loc[data['SMA_short'] > data['SMA_long'], 'Signal'] = 1  # Buy Signal
    data.loc[data['SMA_short'] < data['SMA_long'], 'Signal'] = -1  # Sell Signal
    
    return data

data = moving_average_crossover(data)

plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='Close Price', color='blue')
plt.plot(data.index, data['SMA_short'], label='10-Day SMA', color='orange')
plt.plot(data.index, data['SMA_long'], label='30-Day SMA', color='green')
plt.plot(data[data['Signal'] == 1].index, data['SMA_short'][data['Signal'] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')
plt.plot(data[data['Signal'] == -1].index, data['SMA_short'][data['Signal'] == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')
plt.title('AAPL Moving Average Crossover Strategy (2024-06-01 to 2024-06-03)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
