#! /bin/python3

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import csv
import sys
import getopt
import json


from sklearn import impute
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn import neighbors

def GetImputer(options):
    type = 'simple'
    strategy = 'mean'
    if( 'impute' in options):
        if( 'type' in options['impute']):
            type =  options['impute']['type']
        if( 'strategy' in options['impute']):
            strategy = options['impute']['strategy']

    if( type == 'simple'):
        return impute.SimpleImputer(missing_values=np.nan, strategy=strategy)
    
    return impute.SimpleImputer(missing_values=np.nan, strategy=strategy)

def GetScaler(options):
    type = 'standard'
    if 'scaler' in options:
        if 'type' in options['scaler']:
            type = options['scaler']['type']

    if type == 'standard':
        return preprocessing.StandardScaler()
    if type == 'robust':
        quantileLower = 0.25
        quantileUpper = 0.75
        if 'quantileLower' in options['scaler']:
            quantileLower = options['scaler']['quantileLower']
        if 'quantileUpper' in options['scaler']:
            quantileUpper = options['scaler']['quantileUpper']
        
        return preprocessing.RobustScaler(quantile_range=(quantileLower, quantileUpper))


def help():
    print("task1_automated.py -c [config-file] -o [output file]")


def main():
    # define standard parameters
    outfile = 'prediction.csv'
    config = 'config.json'

    # parse command line args
    try:
        opts, args = getopt.getopt(sys.argv, 'hc:o:', {
            "config=", "output="
        })
    except getopt.GetoptError:
        help()
        sys.exit(2)
    
    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit(0)
        elif opt in {'-o', '--output'}:
            outfile = arg
        elif opt in {'-c', '--config'}:
            config = arg
    
    # load options
    with open(config) as json_file:
        options = json.load(json_file)
    
    # read data
    xtrain = pd.read_csv(
        '../X_train.csv',
        index_col='id',
        dtype={'id':np.int32}
    )
    ytrain = pd.read_csv(
        '../y_train.csv',
        index_col='id', 
        dtype={'id':np.int32})
    xtest = pd.read_csv(
        '../X_test.csv',
        index_col='id',
        dtype={'id':np.int32})

    # impute missing data
    imp = GetImputer(options)
    imp.fit(xtrain)

    xtrain.loc[:,:] = imp.transform(xtrain.values)
    xtest.loc[:,:] = imp.transform(xtest.values)

    lowerVar = 1e-8
    upperVar = 1e13
    if 'variance' in options:
        if 'lower' in options['variance']:
            lowerVar = options['variance']['lower']
        if 'upper' in options['variance']:
            upperVar = options['variance']['upper']
    
    # drop low variance
    thresholder = feature_selection.VarianceThreshold(threshold=lowerVar)
    thresholder.fit(xtrain)

    print("Will drop due to low variance: ", xtrain.columns[ thresholder.get_support()])

    xtrain2 = pd.DataFrame(
        data=thresholder.transform(xtrain),
        index=xtrain.index.tolist(),
        columns=xtrain.columns[ thresholder.get_support()].tolist()
    )

    xtest2 = pd.DataFrame(
        data=thresholder.transform(xtest),
        index=xtest.index.tolist(),
        columns=xtest.columns[ thresholder.get_support()].tolist()
    )

    # drop high variance
    # in default 1e13 this will be 4 features
    var = xtrain2.var()
    tooHigh = var[ var > upperVar ].index.tolist()

    print("Will drop due to high variance: ", tooHigh)

    xtrain3 = xtrain2.drop(columns=tooHigh)
    xtest3 = xtest2.drop(columns=tooHigh)

    #scale features
    scaler = GetScaler(options)
    scaler.fit(xtrain3)

    xtrain3.loc[:,:] = scaler.transform(xtrain3)
    xtest3.loc[:,:] = scaler.transform(xtest3)

    # remove outliers

    X_plot = np.linspace(-3.5, 3.5, 1000)[:, np.newaxis]
    idx= 0
    for col in xtrain3.columns:
        if idx% 10 == 0:
            print("At feature: ", idx)
        
        
        #ax = plt.gca()
        tophat = neighbors.KernelDensity(kernel='tophat').fit( xtrain3[col].values.reshape(-1,1) )
        laplace = neighbors.KernelDensity(kernel='exponential').fit( xtrain3[col].values.reshape(-1,1) )
        gauss = neighbors.KernelDensity(kernel='gaussian').fit( xtrain3[col].values.reshape(-1,1))
        log_dens_th = tophat.score_samples(X_plot)
        log_dens_la = laplace.score_samples(X_plot)
        log_dens_ga = gauss.score_samples(X_plot)
        plt.hist( xtrain3[col] , bins=100, label='hist', density=True, fc=(0,0,1,0.5))
        plt.plot(X_plot[:,0], np.exp(log_dens_th), '-', label='Tophat')
        plt.plot(X_plot[:,0], np.exp(log_dens_la), '-', label='Laplace')
        plt.plot(X_plot[:,0], np.exp(log_dens_ga), '-', label='Gaussian')        
        
        plt.title("Feature {}".format(col))
        plt.legend()
        

        plt.savefig("plots/histogram{:03d}.png".format(idx))
        plt.clf()
        idx = idx+1

        if idx == 20:
            break

        




if __name__ == "__main__":
    main()