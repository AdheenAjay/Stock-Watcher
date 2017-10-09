"""
Created on 20170907_16:12

Desciption:
	This code aims at predicting VOLATILITY S&P 500 (^VIX) time series.
	It reads stock history data and trains a model.
	The model later used to predict future data.
"""
import numpy as np
import time
import pandas as pd
import talib as tal
import datetime
import sklearn
from sklearn.ensemble import RandomForestClassifier

def readData_pandas(in_filePath):

	data = pd.read_csv(in_filePath, index_col = 0, parse_dates = True)
	return data

def prepareData(data, lookBackTimePeriod = 10):
	"""
	Calculate various technical stock measures using TA-Lib lib.
	Measures include..
		RSI : Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.
		SMA : Simple Moving Average
		ADX : The Trend Strength Indicator. Trading in the direction of a strong trend reduces risk and increases profit potential. The average directional index (ADX) is used to determine when price is trending strongly.
		Correlation : 
		Probabilitic SAR :
		Return : return of the past one day on an open to open basis; return = log(openPrice(i)/ openPrice(i-1))
	#lookBackTimePeriod is in days

	"""

	data['high'] = data['High'].shift(1)
	data['low'] = data['Low'].shift(1)
	data['close'] = data['Close'].shift(1)

	data['RSI'] 	= tal.RSI(np.array(data['close']), timeperiod = lookBackTimePeriod)
	data['SMA'] 	= data['close'].rolling(window = lookBackTimePeriod).mean()
	data['ADX'] 	= tal.ADX(np.array(data['high']), np.array(data['low']), np.array(data['close']), timeperiod = lookBackTimePeriod)
	data['Return'] 	= np.log(data['Open'] / data['Open'].shift(1))

	data = data.drop(['High','Low','Close'], axis=1)
	data = data.dropna()

	data['Velocity'] = data['Return']
	data.Velocity[ data.Velocity < 0.] = -1
	data.Velocity[ data.Velocity >= 0.] = 1
	return data

def divideDataForClassification(data, testStartDate):
	features = data.columns[1:-1]
	X = data[features]
	y = data.Velocity

	trainX 	= X[X.index <  testStartDate]
	testX	= X[X.index >= testStartDate]
	if len(testX) <= 5:
		print "[ERROR] There is no data for this test date. Choose an earlier date."

	trainY 	= y[y.index <  testStartDate]
	testY	= y[y.index >= testStartDate]
	return trainX, trainY, testX, testY

def performClassification(trainX, trainY, testX, testY):

	#Random Forest Binary Classification
	clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
	clf.fit(trainX, trainY)
	accuracy_RF = clf.score(testX, testY)
	predicted_RF = clf.predict(testX)

	return predicted_RF, accuracy_RF, clf
	# clf = sklearn.neighbors
	# clf = sklearn.ensemble.AdaBoostClassifier
	# clf = sklearn.ensemble.GradientBoostingClassifier
	# clf = sklearn.svm.SVC

def predictTrend(clf, testDate, testX):
	testDateHistory = testX[testX.index < testDate]
	testDateHistory = testDateHistory[-15:]

	predicted = clf.predict(testDateHistory)

	meanTrend =  predicted.mean()
	if meanTrend > 0:
		return 'Positive'
	else:
		return 'Negative'

def getStockTrend(
		in_filePath, 
		testDate = datetime.datetime(2017, 1, 1),
		train_endDate = datetime.datetime(2016, 12, 1)
	):
	#read data using pandas; as dataframes
	dataSet = readData_pandas(in_filePath)
	# calculate stock valuing measures and append with dataset
	data = prepareData(dataSet)
	#divide data into train and test sets based on train-end-date
	trainX, trainY, testX, testY = divideDataForClassification(data, testStartDate = train_endDate)

	#identify regimes in the stock price
	#Regimes are nothing different states of the specific stock price. (it could be like bullish, bearish or normal)
	predictedArray, accuracy, clf = performClassification(trainX, trainY, testX, testY)
	#predict the trend using the responses in previous 5 days
	predictedTrend = predictTrend(clf,testDate, testX)

	return predictedTrend