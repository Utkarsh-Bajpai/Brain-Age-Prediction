#! /bin/python3

import numpy as np
import pandas as pd
import csv
import sys
import getopt

from sklearn.experimental import enable_iterative_imputer
from sklearn import impute
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn import linear_model
from sklearn import ensemble
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

from scipy import stats    

def help():
    print("attempt3.py args")
    print("arguments:")
    print(" -h --help: print this")
    print(" --lv= x  : columns with variance below x get discarded")
    print(" --uv= x  : columns with variance above x get discarded")
    print(" --uc= x  : entangled featuers with covariance above x get discarded")
    print(" --df= x  : datapoints farther than x times the std deviation from the mean get discarded")
    print(" --of= x  : output file")
    print(" --norm= x: scaler (robust or standard)")

def main():
    # read in data
    xtrain = pd.read_csv(
        '../X_train.csv', 
        index_col='id', 
        dtype={'id':np.int32})
    ytrain = pd.read_csv(
        '../y_train.csv',
        index_col='id', 
        dtype={'id':np.int32})
    xtest = pd.read_csv(
        '../X_test.csv',
        index_col='id',
        dtype={'id':np.int32})

    lowerVar = 1e-8
    upperVar = 1e100
    upperCorr = 0.7
    stdFactor = 2.5
    imputer = 'simple'
    outfile = "prediction"
    norm= 'robust'

        # read parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", [
            "uc=", "df=", "imp=", "of=", "help", "norm="
        ])
    except getopt.GetoptError:
        help()
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in {'-h', '--help'}:
            help()
            sys.exit()
        elif opt == '--uc':
            upperCorr = float(arg)
        elif opt == '--df':
            stdFactor = float(arg)
        elif opt == '--imp':
            imputer = arg
        elif opt == '--of':
            outfile = arg
        elif opt == '--norm':
            norm = arg


    print("Selected parameters:")
    print("  lower variance: {:e}".format(lowerVar))
    print("  upper variance: {:e}".format(upperVar))
    print("  upper covariance: {:e}".format(upperCorr))
    print("  std deviation factor: {:f}".format(stdFactor))
    print("  imputer: ", imputer)
    print("  output: ", outfile)
    print("  norm: ", norm)

    #------------------------------------------------------------------
    # impute
    if imputer == 'simple':
        imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    elif imputer == 'iter':
        imp = impute.IterativeImputer(missing_values=np.nan, max_iter=50, n_nearest_features=20)
    
    imp.fit(xtrain)

    xtrain.loc[:,:] = imp.transform(xtrain)
    xtest.loc[:,:] = imp.transform(xtest)

    #--------------------------------------------------------------------
    # drop features because of variance
    thresholder = VarianceThreshold(threshold=lowerVar)
    thresholder.fit(xtrain)
    print("Will drop because of variance < {:e}: ".format(lowerVar))
    print(xtrain.columns[ np.invert(thresholder.get_support())].tolist())
    
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

    # drop features with absurdly high variance
    var = xtrain2.var()

    tooHigh = var[ var > upperVar].index.tolist() # 4 features
    print("dropping because of variance > {:e}: ".format(upperVar))
    print(tooHigh)

    xtrain3 = xtrain2.drop(columns=tooHigh)
    xtest3 = xtest2.drop(columns=tooHigh)

    #----------------------------------------------------------------------
    # remove outliers
    scaler = RobustScaler()
    scaler.fit(xtrain3)

    xtrain3[ xtrain3 > scaler.center_ + stdFactor*scaler.scale_ ] = np.nan
    xtrain3[ xtrain3 < scaler.center_ - stdFactor*scaler.scale_ ] = np.nan

    xtest3[ xtest3 > scaler.center_ + stdFactor*scaler.scale_ ] = np.nan
    xtest3[ xtest3 < scaler.center_ - stdFactor*scaler.scale_ ] = np.nan

    if imputer == 'simple':
        imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    elif imputer == 'iter':
        imp = impute.IterativeImputer(missing_values=np.nan, max_iter=50, n_nearest_features=20)

    imp.fit(xtrain3)

    xtrain3.loc[:,:] = imp.transform(xtrain3)
    xtest3.loc[:,:]  = imp.transform(xtest3)

    #----------------------------------------------------------------------
    # normalize data
    if norm == 'robust':
        scaler = RobustScaler()
    elif norm == 'standard':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
        
    scaler.fit(xtrain3)

    xtrain3.loc[:,:] = scaler.transform(xtrain3)
    xtest3.loc[:,:] = scaler.transform(xtest3)

    # drop highly correlated features
    corr = xtrain3.corr().abs()
    corr_triu = corr.where( np.triu(np.ones(corr.shape),k=1).astype(np.bool) )
    to_drop = [column for column in corr_triu.columns if any(corr_triu[column] > upperCorr)]

    print("Will drop due to covariance > {:e}: ".format(upperCorr))
    print(to_drop)
    xtrain4 = xtrain3.drop(columns= to_drop)
    xtest4 = xtest3.drop(columns= to_drop)

    print("Using ElasticNet to determine features:")
    netCV = linear_model.ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.75, 0.85],
        alphas=(0., 0.1, 0.2, 0.25, 0.3, 0.35, 0.4),
        cv=5,
        n_jobs=2, max_iter=5e3
    )
    netCV.fit(xtrain4, ytrain.values.ravel())

    print("Selected nr of featrues: ", np.count_nonzero(netCV.coef_))
    print("Selected alpha: ", netCV.alpha_)
    print("Selected l1_ratio: ", netCV.l1_ratio_)
    net = linear_model.ElasticNet(l1_ratio=netCV.l1_ratio_, alpha=netCV.alpha_, max_iter=5e3)

    score = cross_val_score(net, xtrain4, ytrain.values.ravel(), cv=5, scoring='r2')
    print("Score of ElasticNet: ", score)

    net.fit(xtrain4, ytrain.values.ravel())
    print("Selected nr of features after cv: ", np.count_nonzero(net.coef_))

    xtrain5 = xtrain4.loc[:, np.abs(net.coef_) > 0 ]
    xtest5 = xtest4.loc[:, np.abs(net.coef_) > 0 ]

    print("Retained features: ", len(xtest5.columns.tolist()))
    print( xtest5.columns.tolist() )

    print("Testing new elasticNet:")
    netCV2 = linear_model.ElasticNetCV(
        l1_ratio=[0., 0.1, 0.3, 0.5, 0.75, 0.85, 0.9, 1.],
        alphas=(0., 0.1, 0.2, 0.25, 0.3, 0.35, 0.4),
        cv=5,
        n_jobs=2, max_iter=5e3
    )
    netCV2.fit(xtrain5, ytrain.values.ravel())

    print("Selected nr of features: ", np.count_nonzero(netCV2.coef_))
    print("Selected alpha: ", netCV2.alpha_)
    print("Selected l1_ration: ", netCV2.l1_ratio_)

    net2 = linear_model.ElasticNet(l1_ratio=netCV2.l1_ratio_, alpha=netCV2.alpha_, max_iter=5e3)
    score = cross_val_score(net2, xtrain5, ytrain.values.ravel(), cv=5, scoring='r2')
    print("Score of new elasticNet: ", score)
    net2.fit(xtrain5, ytrain.values.ravel())
    print("Would use: ", np.count_nonzero(net2.coef_), " features")

if __name__ == "__main__":
    main()