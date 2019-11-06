from backtester.trading_system_parameters import TradingSystemParameters
from backtester.features.feature import Feature
from datetime import datetime, timedelta
from backtester.trading_system import TradingSystem
from backtester.version import updateCheck
from auquan_qq1_toolbox.problem1_trading_params import MyTradingParams

from sklearn import linear_model
from sklearn import metrics as sm
import pandas as pd
import numpy as np
import sys

## Make your changes to the functions below.
## SPECIFY the symbols you are modeling for in getSymbolsToTrade() below
## You need to specify features you want to use in getInstrumentFeatureConfigDicts() and getMarketFeatureConfigDicts()
## and create your predictions using these features in getPrediction()

## Don't change any other function
## The toolbox does the rest for you, from downloading and loading data to running backtest


class MyTradingFunctions():

    def __init__(self):  #TODO 0: Put any global variables here
        self.lookback = 1300  ## max number of historical datapoints you want at any given time
            
        self.params = {}
        # for example you can import and store an ML model from scikit learn in this dict
        # or train it on the go
        self.model = {}
        # and set the frequency of data points at which you want to update the model
        self.updateFrequency = 130


    ###########################################
    ## ONLY FILL THE FOUR FUNCTIONS BELOW    ##
    ###########################################

    ###############################################################################
    ### TODO 1: FILL THIS FUNCTION TO specify all stockIDs you are modeling for ###
    ### USE TEMPLATE BELOW AS EXAMPLE                                           ###
    ###############################################################################

    def getSymbolsToTrade(self):
        return ['SIZ', 'MLQ']

    '''
    Specify all Features you want to use by  by creating config dictionaries.
    Create one dictionary per feature and return them in an array.
    Feature config Dictionary have the following keys:
        featureId: a str for the type of feature you want to use
        featureKey: {optional} a str for the key you will use to call this feature
                    If not present, will just use featureId
        params: {optional} A dictionary with which contains other optional params if needed by the feature
    msDict = {'featureKey': 'ms_5',
              'featureId': 'moving_sum',
              'params': {'period': 5,
                         'featureName': 'basis'}}
    return [msDict]
    You can now use this feature by in getPRediction() calling it's featureKey, 'ms_5'
    '''

    def getInstrumentFeatureConfigDicts(self):

    ##############################################################################
    ### TODO 2a: FILL THIS FUNCTION TO CREATE DESIRED FEATURES for each symbol. ###
    ### USE TEMPLATE BELOW AS EXAMPLE                                          ###
    ##############################################################################
        mom1Dict = {'featureKey': 'mom_5',
                   'featureId': 'momentum',
                   'params': {'period': 5,
                              'featureName': 'F5'}}
        mom2Dict = {'featureKey': 'mom_10',
                   'featureId': 'momentum',
                   'params': {'period': 10,
                              'featureName': 'F5'}}
        ma1Dict = {'featureKey': 'ma_5',
                   'featureId': 'moving_average',
                   'params': {'period': 5,
                              'featureName': 'F5'}}
        ma2Dict = {'featureKey': 'ma_10',
                   'featureId': 'moving_average',
                   'params': {'period': 10,
                              'featureName': 'F5'}}
        return [mom1Dict, mom2Dict, ma1Dict, ma2Dict]



    def getMarketFeatureConfigDicts(self):
    ###############################################################################
    ### TODO 2b: FILL THIS FUNCTION TO CREATE features that use multiple symbols ###
    ### USE TEMPLATE BELOW AS EXAMPLE                                           ###
    ###############################################################################

        # customFeatureDict = {'featureKey': 'custom_mrkt_feature',
        #                      'featureId': 'my_custom_mrkt_feature',
        #                      'params': {'param1': 'value1'}}
        return []

    '''
    Combine all the features to create the desired 0/1 predictions for each symbol.
    'predictions' is Pandas Series with symbol as index and predictions as values
    The holder for all the instrument features is lookbackInstrumentFeatures
    To load a dataframe for a feature using its feature_key simply do:
        ms5Data = lookbackInstrumentFeatures.getFeatureDf('ms_5')
    This returns a dataFrame for that feature for ALL symbols for all times upto lookback time
    Now you can call just the last data point for ALL symbols as
        ms5 = ms5Data.iloc[-1]
    You can call last datapoint for one symbol 'ABC' as
        value_for_abs = ms5['ABC']
    Output of the prediction function is used by the toolbox to make further trading decisions and evaluate your score.
    '''


    def getPrediction(self, time, updateNum, lookbackInstrumentFeatures, lookbackMarketFeatures, predictions, targetVariable):

        #############################################################################################
        ###  TODO 3 : FILL THIS FUNCTION TO RETURN A 0/1 prediction for each stockID              ###
        ###  You can use all the features created above and combine then using any logic you like ###
        ###  USE TEMPLATE BELOW AS EXAMPLE                                                        ###
        #############################################################################################

        # if you don't enough data yet, don't make a prediction
        if updateNum<=2*self.updateFrequency:
            return predictions

        # Once you have enough data, start making predictions

        # Loading the target Variable
        Y = lookbackInstrumentFeatures.getFeatureDf(targetVariable)

        # Loading features

        featureList = lookbackInstrumentFeatures.getAllFeatures()

        mom1 = lookbackInstrumentFeatures.getFeatureDf('mom_5')     #DF with rows=timestamp and columns=stockIDS
        mom2 = lookbackInstrumentFeatures.getFeatureDf('mom_10')    #DF with rows=timestamp and columns=stockIDS
        factor1Values = (mom1/mom2)                                 #DF with rows=timestamp and columns=stockIDS
        
        ma1 = lookbackInstrumentFeatures.getFeatureDf('ma_5')       #DF with rows=timestamp and columns=stockIDS
        ma2 = lookbackInstrumentFeatures.getFeatureDf('ma_10')      #DF with rows=timestamp and columns=stockIDS
        factor2Values = (ma1/ma2)                                   #DF with rows=timestamp and columns=stockIDS

        # Now looping over all stocks:
        for s in self.getSymbolsToTrade():
            #Creating a dataframe to hold features for this stock
            X = pd.DataFrame(index=Y.index, columns=featureList)         #DF with rows=timestamp and columns=featureNames
            for f in featureList:
                X[f] = lookbackInstrumentFeatures.getFeatureDf(f)
            X['F1'] = factor1Values[s]
            X['F2'] =factor2Values[s]

            # if this is the first time we are training a model, start by creating a new model
            if s not in self.model:
                self.model[s] = linear_model.LogisticRegression()

            # we will update this model during further runs

            # if you are at the update frequency, update the model
            if (updateNum-1)%self.updateFrequency==0:

                # drop nans and infs from X
                X = X.replace([np.inf, -np.inf], np.nan).dropna()
                # create a target variable vector for this stock, with same index as X
                y_s = Y[s].loc[Y.index.isin(X.index)]

                print('Training...')
                # make numpy arrays with the right shape
                x_train = np.array(X)[:-1]                         # shape = timestamps x numFeatures
                y_train = np.array(y_s)[:-1].astype(int).reshape(-1) # shape = timestamps x 1
                self.model[s].fit(x_train, y_train)

            # make your prediction using your model
            # first verify none of the features are nan or inf
            if X.iloc[-1].replace([np.inf, -np.inf], np.nan).hasnans:
                print('Test Feature Data has nans')
                y_predict = 0.5
            else:
                y_predict = self.model[s].predict(X.iloc[-1].values.reshape(1,-1))

            # if you are making probabilistic predictions, set a threshold to convert them to 0/1
            threshold = 0.8
            predictions[s] = 1 if y_predict>threshold else 0.5
            predictions[s] = 0 if y_predict<(1-threshold) else 0.5

        return predictions

    ###########################################
    ##         DONOT CHANGE THESE            ##
    ###########################################

    def getLookbackSize(self):
        return self.lookback

    ###############################################
    ##  CHANGE ONLY IF YOU HAVE CUSTOM FEATURES  ##
    ###############################################

    def getCustomFeatures(self):
        return {'my_custom_feature_identifier': MyCustomFeatureClassName}

####################################################
##   YOU CAN DEFINE ANY CUSTOM FEATURES HERE      ##
##  If YOU DO, MENTION THEM IN THE FUNCTION ABOVE ##
####################################################
class MyCustomFeatureClassName(Feature):
    ''''
    Custom Feature to implement for instrument. This function would return the value of the feature you want to implement.
    1. create a new class MyCustomFeatureClassName for the feature and implement your logic in the function computeForInstrument() -
    2. modify function getCustomFeatures() to return a dictionary with Id for this class
        (follow formats like {'my_custom_feature_identifier': MyCustomFeatureClassName}.
        Make sure 'my_custom_feature_identifier' doesnt conflict with any of the pre defined feature Ids
        def getCustomFeatures(self):
            return {'my_custom_feature_identifier': MyCustomFeatureClassName}
    3. create a dict for this feature in getInstrumentFeatureConfigDicts() above. Dict format is:
            customFeatureDict = {'featureKey': 'my_custom_feature_key',
                                'featureId': 'my_custom_feature_identifier',
                                'params': {'param1': 'value1'}}
    You can now use this feature by calling it's featureKey, 'my_custom_feature_key' in getPrediction()
    '''
    @classmethod
    def computeForInstrument(cls, updateNum, time, featureParams, featureKey, instrumentManager):
        # Custom parameter which can be used as input to computation of this feature
        param1Value = featureParams['param1']

        # A holder for the all the instrument features
        lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()

        # dataframe for a historical instrument feature (basis in this case). The index is the timestamps
        # atmost upto lookback data points. The columns of this dataframe are the symbols/instrumentIds.
        lookbackInstrumentValue = lookbackInstrumentFeatures.getFeatureDf('symbolVWAP')

        # The last row of the previous dataframe gives the last calculated value for that feature (basis in this case)
        # This returns a series with symbols/instrumentIds as the index.
        currentValue = lookbackInstrumentValue.iloc[-1]

        if param1Value == 'value1':
            return currentValue * 0.1
        else:
            return currentValue * 0.5


if __name__ == "__main__":
    if updateCheck():
        print('Your version of the auquan toolbox package is old. Please update by running the following command:')
        print('pip install -U auquan_toolbox')
    else:
        print('Loading your config dicts and prediction function')
        tf = MyTradingFunctions()
        print('Loaded config dicts and prediction function, Loading Problem 1 Params')
        tsParams = MyTradingParams(tf)
        print('Loaded Problem 1 Params, Loading Backtester and Data')
        tradingSystem = TradingSystem(tsParams)
        print('Loaded Backtester and Data Loaded, Backtesting')
    # Set onlyAnalyze to True to quickly generate csv files with all the features
    # Set onlyAnalyze to False to run a full backtest
    # Set makeInstrumentCsvs to False to not make instrument specific csvs in runLogs. This improves the performance BY A LOT
        tradingSystem.startTrading(onlyAnalyze=False, shouldPlot=True, makeInstrumentCsvs=True)
