import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_data(symbol, date):
    start_datetime = datetime.combine(date, datetime.min.time())
    end_datetime = start_datetime + timedelta(days=1)
    tickerData = yf.Ticker(symbol)
    intraday_data = tickerData.history(start=start_datetime.strftime('%Y-%m-%d'),
                                       end=end_datetime.strftime('%Y-%m-%d'),
                                       interval='1m')  
    if (intraday_data.empty) : return intraday_data
    market_start, market_end = "09:15", "15:30"
    intraday_data = intraday_data.between_time(market_start, market_end)

    return intraday_data #pd dataframe object