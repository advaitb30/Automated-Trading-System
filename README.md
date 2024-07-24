Task1 was reading up the basics of stock market and indicators used to make trades

Task2 includes the python code for a simple trading bot. We used the following python libraries: Numpy, Pandas, matplotlib(for plotting the data and indicators), dates and an API library yfinance(Yahoo finance).

Task 3 was exploring ML, through simple regression models on PyTorch.

Task 4. I created a basic Neural Network which outputs a buy(1) or sell(0) signal based on indicators. For training the model I used a dataset from Kaggle which contains 3k+ entries of the NIFTY 50 index of frequency '1d'. The accuracy of the model comes to about 58% which isn't great per normal standards, but given the simplicity of the mode, it is a great accuracy. The features used were: MACD line, EMA50, OBV, Signal line, percentage change, RSI, etc. Only buy and sell options were kept. If there was a hold option as well as features that keep a track of profit, accuracy should increase. Also making trades when probability is above a certain threshold will greatly improve accuracy. Overall the final task was very insightful as a got a great understanding of Pandas and applied Neural Networks which was really fun! 
