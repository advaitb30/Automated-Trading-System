import numpy as np
import pandas as pd

df = pd.read_csv("NSEI .csv") #this dataset contains entries of the NIFTY 50 index of frequency 1d
df.dropna(subset = ['Close'], inplace = True) #drops entries with empty values (checks column ['Close'])
df = df.reset_index(drop = 'True') #this commands resets the indexes
prices = df['Close']
#print(df.head()) can be used to check first 5 entries
print(prices)

def calculate_rsi(prices, Window): #calculates the indicator RSI
    delta = prices.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window = Window).mean()
    avg_loss = loss.rolling(window = Window).mean()
    rs = avg_gain/avg_loss
    rsi = 100 - 100/(1 + rs)
    return rsi

def calculate_ema(prices, start, Window): #calculates the ema for a price
    ema = prices.iloc[start-Window]
    multiplier = 2/(Window + 1)
    for i in range (start - Window, start) :
        ema = multiplier*prices.iloc[i] + ema*(1 - multiplier)
        #print(f"E={ema[i]} index={i}")
    return ema 

def series_ema(prices, Window): #Applies the ema formula on a series of prices (any Series it inputs)
    ema = pd.Series()
    for i in range(len(df)):
        if i < Window:
            ema[i] = None
        else:
            ema[i] = calculate_ema(prices, i, Window)
    return ema

def calculate_moving_averages(prices, short_window = 12, long_window = 26): #calculates moving average (short and long ema)
    #short_sma = prices.rolling(window = short_window).mean()
    #long_sma = prices.rolling(window = long_window).mean()
    short_ema = series_ema(df['Close'], 12)
    long_ema = series_ema(df['Close'], 26)
    df['short_ema'] = short_ema
    df['long_ema'] = long_ema
    df.dropna(subset = ['long_ema'], inplace = True)
    return short_ema, long_ema

short_ema, long_ema = calculate_moving_averages(prices)
MACD = df['short_ema'] - df['long_ema'] #formula of MACD
df['MACD'] = MACD #inserts a column for MACD
signal_line = series_ema(df['MACD'], 9)
df['SL'] = signal_line #inserts acolumn for signal line
df.dropna(subset = ['SL'], inplace = True) #cleans the data, getting rid of entries with NULL values in signal line column
df = df.reset_index(drop = 'True')
print(df)

def calculate_pc(prices): #calculates percentage chane from previous day's closing to current day's closing
    cpc = pd.Series()
    for i in range(len(df)):
        if i == 0:
            cpc[i] = None
        else:
            cpc[i] = ((prices.iloc[i] - prices.iloc[i-1])/prices.iloc[i-1])*100
    return cpc

def calculate_obv(): #calculates the "on-balance volume indicator"
    obv = pd.Series()
    for i in range(len(df)):
        if i == 0:
            obv[i] = df.loc[i, 'Volume']
        else:
            if(df.loc[i, 'Close'] > df.loc[i-1, 'Close']):
                obv[i] = obv.iloc[i-1] + df.loc[i, 'Volume']
            elif(df.loc[i, 'Close'] < df.loc[i-1, 'Close']):
                obv[i] = obv.iloc[i-1] - df.loc[i, 'Volume']
            else:
                obv[i] = obv.iloc[i-1]
    return obv

def calculate_target(): #target value(y) contains binary values for either buying or selling()
    target = pd.Series()
    for i in range(len(df) - 1):
            if(df.loc[i+1, 'Close'] > df.loc[i, 'Close']):
                target[i] = 1
            else:
                target[i] = 0
    return target

def williamsR(period): #calculates the indicator williamsR
    k = pd.Series()
    for i in range(len(df)):
        if i < 26:
            k[i] = 0
        else:
            current = df.loc[i,'Close']
            high_period = df.loc[i-period:i, 'High'].max()
            low_period = df.loc[i-period:i, 'Low'].min()
            k[i] = (current - high_period)/(high_period - low_period)
    return k

RSI = calculate_rsi(df['Close'], 14)
df['RSI'] = RSI #inserts a column for MACD
EMA50 = series_ema(df['Close'], 50)
df['ema50'] = EMA50 #inserts a column for EMA50w
CO = df['Close'] - df['Open']
df['close-open'] = CO #inserts a column for close-open difference for current day's price
HL = df['High'] - df['Low']
df['High- Low'] = HL #inserts a column for High-Low difference in current day's price
k = williamsR(14)
df['WR'] = k #inserts a column for WilliamsR

print(df)

CPC = calculate_pc(df['Close'])
OBV = calculate_obv()
y = calculate_target()
df['cpc'] = CPC #inserts a column for percentage change
df['obv'] = OBV #inserts a column for obv
df['Y'] = y #inserts a column for target value(Y)

#normalisation of features
df['normMACD'] = (df['MACD'] - df['MACD'].mean())/df['MACD'].std()
df['normSL'] = (df['SL'] - df['SL'].mean())/df['SL'].std()
df['RSI'] = df['RSI']/10
df['close-open'] = (df['close-open'] - df['close-open'].min())/(df['close-open'].max() - df['close-open'].min())
df['High- Low'] = (df['High- Low'] - df['High- Low'].min())/(df['High- Low'].max() - df['High- Low'].min())
df['obv'] = (df['obv'] - df['obv'].min())/(df['obv'].max() - df['obv'].min())
df['ema50'] = (df['ema50'] - df['ema50'].mean())/(df['ema50'].std())
df['WR'] = (df['WR'] - df['WR'].mean())/(df['WR'].std())

df.dropna(subset = ['ema50'], inplace = True) #drops all the entries with NULL values in the column 'ema50'
df = df.reset_index(drop = True)
print(df)

features = ['RSI', 'ema50', 'close-open', 'High- Low', 'WR', 'cpc', 'normMACD', 'normSL']
X = df[features]
Y = df['Y']

#splitting features on a 80-20 basis
X_train = X.loc[:2572]
Y_train = Y.loc[:2572]
X_test = X.loc[2573:]
Y_test = Y.loc[2573:]

#creating the neural network using tensorflow(keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
tf.random.set_seed(1234)
model = Sequential(
[
    tf.keras.Input(shape = (8, )),
    Dense(units = 120, activation = 'linear', name = 'Layer1'),
    Dense(units = 40, activation = 'linear', name = 'Layer2'),
    Dense(units = 4, activation = 'linear', name = 'Layer3'),
    Dense(units = 1, activation = 'sigmoid', name = 'Output')
])
model.compile(
    loss = tf.keras.losses.MeanSquaredError(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.003),
    metrics = [tf.keras.metrics.Accuracy()]
)

model.fit(
    X_train, Y_train,            
    epochs = 150,
)

predict = model.predict(X_test)
Y_test.reset_index(drop = True)
print(Y_test)
for i in range(len(predict)):
    if(predict[i] > 0.5):
        predict[i] = 1
    else:
        predict[i] = 0
count = 0
for j in range(len(Y_test)):
    if(Y_test[j+2573] == predict[j]):
        count += 1
print(count/len(Y_test))