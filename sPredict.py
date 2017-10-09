"""
Created on 20170907_16:07

Desciption:
	This code aims at predicting VOLATILITY S&P 500 (^VIX) time series.
	It reads stock history data and trains a model.
	The model later used to predict future data.
"""

import numpy as np
import datetime
import stockPredictor as sPred

####################################################
#c o n f i g  d a t a

# data file path
in_stockDataPath = '^VIX.csv'

if __name__ == "__main__":
	
	#call function to calculate stock-price-trend
	trend = sPred.getStockTrend(in_stockDataPath, testDate = datetime.datetime(2017, 5, 23))
	
	print "\n\nStockWatcher Prediction: "
	print "The stock is showing [%s] trend.!" %trend
