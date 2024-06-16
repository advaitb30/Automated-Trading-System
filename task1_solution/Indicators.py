import numpy as np
import pandas as pd
import yfinance as yf
from fetchdata import fetch_data
import matplotlib.pyplot as plt
ticker = "ITC.NS"
intraday_data = fetch_data(ticker, )
prices = intraday_data['Close']
def calculate_moving_averages(prices, short_window = 14, long_window = 26):
    short_ma = prices.rolling(window = short_window)
    long_ma = prices.rolling(window = long_window)
    macd = short_ma - long_ma
    return short_ma, long_ma
    
def calculate_rsi(prices, Window):
    delta = prices.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window = Window).mean()
    avg_loss = loss.rolling(window = Window).mean()
    rs = avg_gain.to_numpy/avg_loss.to_numpy
    rsi = 100 - 100/(1 + rs)
    return rsi
