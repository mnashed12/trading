import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download historical data for a given ticker (e.g., Apple)
data = yf.download('AAPL', start='2020-01-01', end='2024-01-01')

# Fill missing values
data.fillna(method='ffill', inplace=True)

# Calculate moving averages
data['SMA50'] = data['Close'].rolling(window=50).mean()
data['SMA200'] = data['Close'].rolling(window=200).mean()

# Initial capital for backtesting
initial_capital = 10000

def moving_average_crossover(data):
    buy_signals = []
    sell_signals = []
    position = False

    for i in range(len(data)):
        if data['SMA50'][i] > data['SMA200'][i] and not position:
            buy_signals.append(data['Close'][i])
            sell_signals.append(None)
            position = True
        elif data['SMA50'][i] < data['SMA200'][i] and position:
            buy_signals.append(None)
            sell_signals.append(data['Close'][i])
            position = False
        else:
            buy_signals.append(None)
            sell_signals.append(None)

    data['Buy'] = buy_signals
    data['Sell'] = sell_signals
    return data

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def rsi_strategy(data, rsi_period=14, buy_threshold=30, sell_threshold=70):
    data['RSI'] = calculate_rsi(data, rsi_period)
    buy_signals = []
    sell_signals = []
    position = False

    for i in range(len(data)):
        if data['RSI'][i] < buy_threshold and not position:
            buy_signals.append(data['Close'][i])
            sell_signals.append(None)
            position = True
        elif data['RSI'][i] > sell_threshold and position:
            buy_signals.append(None)
            sell_signals.append(data['Close'][i])
            position = False
        else:
            buy_signals.append(None)
            sell_signals.append(None)

    data['Buy_RSI'] = buy_signals
    data['Sell_RSI'] = sell_signals
    return data

def bollinger_bands(data, window=20, num_std_dev=2):
    data['Middle Band'] = data['Close'].rolling(window=window).mean()
    data['Upper Band'] = data['Middle Band'] + num_std_dev * data['Close'].rolling(window=window).std()
    data['Lower Band'] = data['Middle Band'] - num_std_dev * data['Close'].rolling(window=window).std()

    buy_signals = []
    sell_signals = []
    position = False

    for i in range(len(data)):
        if data['Close'][i] < data['Lower Band'][i] and not position:
            buy_signals.append(data['Close'][i])
            sell_signals.append(None)
            position = True
        elif data['Close'][i] > data['Upper Band'][i] and position:
            buy_signals.append(None)
            sell_signals.append(data['Close'][i])
            position = False
        else:
            buy_signals.append(None)
            sell_signals.append(None)

    data['Buy_BB'] = buy_signals
    data['Sell_BB'] = sell_signals
    return data

def macd(data, fast_period=12, slow_period=26, signal_period=9):
    data['MACD'] = data['Close'].ewm(span=fast_period, adjust=False).mean() - data['Close'].ewm(span=slow_period, adjust=False).mean()
    data['Signal Line'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()

    buy_signals = []
    sell_signals = []
    position = False

    for i in range(len(data)):
        if data['MACD'][i] > data['Signal Line'][i] and not position:
            buy_signals.append(data['Close'][i])
            sell_signals.append(None)
            position = True
        elif data['MACD'][i] < data['Signal Line'][i] and position:
            buy_signals.append(None)
            sell_signals.append(data['Close'][i])
            position = False
        else:
            buy_signals.append(None)
            sell_signals.append(None)

    data['Buy_MACD'] = buy_signals
    data['Sell_MACD'] = sell_signals
    return data

def momentum_strategy(data, period=10):
    data['Momentum'] = data['Close'].diff(period)
    
    buy_signals = []
    sell_signals = []
    position = False

    for i in range(len(data)):
        if data['Momentum'][i] > 0 and not position:
            buy_signals.append(data['Close'][i])
            sell_signals.append(None)
            position = True
        elif data['Momentum'][i] < 0 and position:
            buy_signals.append(None)
            sell_signals.append(data['Close'][i])
            position = False
        else:
            buy_signals.append(None)
            sell_signals.append(None)

    data['Buy_Momentum'] = buy_signals
    data['Sell_Momentum'] = sell_signals
    return data

def backtest(data, buy_column, sell_column, initial_capital=10000):
    capital = initial_capital
    position = 0
    for i in range(len(data)):
        if data[buy_column][i] is not None:
            position = capital / data['Close'][i]
            capital = 0
        elif data[sell_column][i] is not None:
            capital = position * data['Close'][i]
            position = 0
    
    final_value = capital if capital > 0 else position * data['Close'][-1]
    return final_value

def calculate_performance_metrics(initial_capital, final_value):
    net_profit = final_value - initial_capital
    roi = (net_profit / initial_capital) * 100
    return net_profit, roi

# Apply the strategies
data = moving_average_crossover(data)
data = rsi_strategy(data)
data = bollinger_bands(data)
data = macd(data)
data = momentum_strategy(data)


# Backtest the strategies
final_value_ma = backtest(data, 'Buy', 'Sell', initial_capital)
final_value_rsi = backtest(data, 'Buy_RSI', 'Sell_RSI', initial_capital)
final_value_bb = backtest(data, 'Buy_BB', 'Sell_BB', initial_capital)
final_value_macd = backtest(data, 'Buy_MACD', 'Sell_MACD', initial_capital)
final_value_momentum = backtest(data, 'Buy_Momentum', 'Sell_Momentum', initial_capital)

# Print results
print(f"Initial Capital: ${initial_capital}")
print(f"Final Portfolio Value (Moving Average): ${final_value_ma}")
print(f"Final Portfolio Value (RSI): ${final_value_rsi}")
print(f"Final Portfolio Value (Bollinger Bands): ${final_value_bb}")
print(f"Final Portfolio Value (MACD): ${final_value_macd}")
print(f"Final Portfolio Value (Momentum): ${final_value_momentum}")
print(f"Net Profit (Moving Average): ${final_value_ma - initial_capital}")
print(f"Net Profit (RSI): ${final_value_rsi - initial_capital}")
print(f"Net Profit (Bollinger Bands): ${final_value_bb - initial_capital}")
print(f"Net Profit (MACD): ${final_value_macd - initial_capital}")
print(f"Net Profit (Momentum): ${final_value_momentum - initial_capital}")


plt.figure(figsize=(14, 10))

# Plot Moving Average Crossover signals
plt.subplot(3, 2, 1)
plt.plot(data['Close'], label='Close Price', alpha=0.5)
plt.plot(data['SMA50'], label='50-day SMA', alpha=0.5)
plt.plot(data['SMA200'], label='200-day SMA', alpha=0.5)
plt.scatter(data.index, data['Buy'], label='Buy Signal (MA)', marker='^', color='green')
plt.scatter(data.index, data['Sell'], label='Sell Signal (MA)', marker='v', color='red')
plt.title('Moving Average Crossover Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Plot RSI signals
plt.subplot(3, 2, 2)
plt.plot(data['Close'], label='Close Price', alpha=0.5)
plt.plot(data['RSI'], label='RSI', alpha=0.5, color='orange')
plt.scatter(data.index, data['Buy_RSI'], label='Buy Signal (RSI)', marker='^', color='green')
plt.scatter(data.index, data['Sell_RSI'], label='Sell Signal (RSI)', marker='v', color='red')
plt.title('RSI Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Plot Bollinger Bands signals
plt.subplot(3, 2, 3)
plt.plot(data['Close'], label='Close Price', alpha=0.5)
plt.plot(data['Middle Band'], label='Middle Band', alpha=0.5, color='purple')
plt.plot(data['Upper Band'], label='Upper Band', alpha=0.5, color='blue')
plt.plot(data['Lower Band'], label='Lower Band', alpha=0.5, color='blue')
plt.scatter(data.index, data['Buy_BB'], label='Buy Signal (BB)', marker='^', color='green')
plt.scatter(data.index, data['Sell_BB'], label='Sell Signal (BB)', marker='v', color='red')
plt.title('Bollinger Bands Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Plot MACD signals
plt.subplot(3, 2, 4)
plt.plot(data['Close'], label='Close Price', alpha=0.5)
plt.plot(data['MACD'], label='MACD', alpha=0.5, color='blue')
plt.plot(data['Signal Line'], label='Signal Line', alpha=0.5, color='red')
plt.scatter(data.index, data['Buy_MACD'], label='Buy Signal (MACD)', marker='^', color='green')
plt.scatter(data.index, data['Sell_MACD'], label='Sell Signal (MACD)', marker='v', color='red')
plt.title('MACD Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Plot Momentum signals
plt.subplot(3, 2, 5)
plt.plot(data['Close'], label='Close Price', alpha=0.5)
plt.plot(data['Momentum'], label='Momentum', alpha=0.5, color='brown')
plt.scatter(data.index, data['Buy_Momentum'], label='Buy Signal (Momentum)', marker='^', color='green')
plt.scatter(data.index, data['Sell_Momentum'], label='Sell Signal (Momentum)', marker='v', color='red')
plt.title('Momentum Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

plt.tight_layout()
plt.show()
