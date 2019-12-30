#! /bin/python3

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import csv
import sys
import getopt

from sklearn.impute import SimpleImputer
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

def printGaussianLikelihood():
    xtrain = pd.read_csv(
        '../X_train.csv', 
        index_col='id', 
        dtype={'id':np.int32})

    
    llh = []
    for col in xtrain.columns:
        data = xtrain[col].values
        data = data[ ~np.isnan(data)]
        mean, std = stats.norm.fit(data)
        if std < 1e-8:
            llh.append(1e4)

        else:
            llh.append(stats.norm.logpdf(data, mean, std).sum())

    plt.hist(llh, bins=100)
    plt.title("Loglikelihoods gaussian")
    plt.show()

def KeepGaussianFeatures():
    xtrain = pd.read_csv(
        '../X_train.csv', 
        index_col='id', 
        dtype={'id':np.int32})
    ytrain = pd.read_csv(
        '../y_train.csv',
        index_col='id', 
        dtype={'id':np.int32})

    
    llh = []
    for col in xtrain.columns:
        data = xtrain[col].values
        data = data[ ~np.isnan(data)]
        mean, std = stats.norm.fit(data)
        if std < 1e-8:
            llh.append(1e5)
        else:
            llh.append(stats.norm.logpdf(data, mean, std).sum())
    
    # get half with best fit to gaussian
    llharr = np.array(llh)
    med = np.median(llharr)

    xtrain2 = xtrain.loc[:, xtrain.columns[llharr < med ].tolist()]

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(xtrain2)

    xtrain2.loc[:,:] = imp.transform(xtrain2)


    scaler = RobustScaler()
    scaler.fit(xtrain2)

    stdFactor = 1.8
    xtrain2[ xtrain2 > scaler.center_ + stdFactor*scaler.scale_ ] = np.nan
    xtrain2[ xtrain2 < scaler.center_ - stdFactor*scaler.scale_ ] = np.nan

    

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(xtrain2)

    xtrain2.loc[:,:] = imp.transform(xtrain2)

    scaler = StandardScaler()
    scaler.fit(xtrain2)

    xtrain2.loc[:,:] = scaler.transform(xtrain2)

    # drop highly correlated features
    upperCovar = 0.8
    corr = xtrain2.corr().abs()
    corr_triu = corr.where( np.triu(np.ones(corr.shape),k=1).astype(np.bool) )
    to_drop = [column for column in corr_triu.columns if any(corr_triu[column] > upperCovar)]

    print("Will drop due to covariance > {:e}: ".format(upperCovar))
    print(to_drop)
    xtrain4 = xtrain2.drop(columns= to_drop)

    print()
    print("Using ElasticNet to determine features:")
    netCV = linear_model.ElasticNetCV(l1_ratio=[0., 0.1, 0.5, 0.75, 0.85, 0.9, 0.95, 1.],
    cv=5, n_jobs=2, max_iter=5e3, alphas=(0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.5))
    netCV.fit(xtrain4, ytrain.values.ravel())

    print("Selected nr of features: ", np.count_nonzero( netCV.coef_))
    print("Selected alpha: ", netCV.alpha_)
    print("Selected l1_ratio: ", netCV.l1_ratio_)
    net = linear_model.ElasticNet(l1_ratio=netCV.l1_ratio_, max_iter=5e3)
    print("Testing ElasticNet:")
    score = cross_val_score(net, xtrain4, ytrain.values.ravel(), cv=5, scoring='r2')
    print("score: ", score)
    net.fit(xtrain4, ytrain.values.ravel())

    xtrain5 = xtrain4.loc[:, net.coef_ > 0 ]

    print()
    print("Testing various regressors....")

    print("Testing ridge regression:")
    ridgeCV = linear_model.RidgeCV(alphas=np.linspace(0,3, 31).tolist(),
    cv=5, scoring='r2')
    ridgeCV.fit(xtrain5, ytrain.values.ravel())

    print("Selected alpha: ", ridgeCV.alpha_)

    ridge = linear_model.Ridge(alpha=ridgeCV.alpha_)
    score = cross_val_score(ridge, xtrain5, ytrain.values.ravel(), cv=5, scoring='r2')
    print("score: ", score)

    print("Testing RanodmForestRegressor:")
    rfr100 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=15)
    score = cross_val_score(rfr100, xtrain5, ytrain.values.ravel(), cv=5, scoring='r2')
    print("For 100 Trees:")
    print("score: ", score)

    rfr500 = ensemble.RandomForestRegressor(n_estimators=500, max_depth=15)
    score = cross_val_score(rfr500, xtrain5, ytrain.values.ravel(), cv=5, scoring='r2')
    print("For 500 Trees:")
    print("score: ", score)

    rfr1000 = ensemble.RandomForestRegressor(n_estimators=1000, max_depth=15)
    score = cross_val_score(rfr1000, xtrain5, ytrain.values.ravel(), cv=5, scoring='r2')
    print("For 1000 Trees:")
    print("score: ", score)
    

        



    

def help():
    print("attempt3.py args")
    print("arguments:")
    print(" -h --help: print this")
    print(" --lv= x  : columns with variance below x get discarded")
    print(" --uv= x  : columns with variance above x get discarded")
    print(" --uc= x  : entangled featuers with covariance above x get discarded")
    print(" --df= x  : datapoints farther than x times the std deviation from the mean get discarded")

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

    # read parameters
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", [
            "lv=", "uv=", "uc=", "df=", "help"
        ])
    except getopt.GetoptError:
        help()
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in {'-h', '--help'}:
            help()
            sys.exit()
        elif opt == '--lv':
            lowerVar = float(arg)
        elif opt == '--uv':
            upperVar = float(arg)
        elif opt == '--uc':
            upperCorr = float(arg)
        elif opt == '--df':
            stdFactor = float(arg)


    print("Selected parameters:")
    print("  lower variance: {:e}".format(lowerVar))
    print("  upper variance: {:e}".format(upperVar))
    print("  upper covariance: {:e}".format(upperCorr))
    print("  std deviation factor: {:f}".format(stdFactor))


    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(xtrain)

    xtrain.loc[:,:] = imp.transform(xtrain)
    xtest.loc[:,:] = imp.transform(xtest)

    # drop features with no in-feature variance
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
    #xtrain3 = xtrain2
    #xtest3 = xtest2

    # remove outliers
    #scaler = StandardScaler()
    scaler = RobustScaler()
    scaler.fit(xtrain3)

    xtrain3[ xtrain3 > scaler.center_ + stdFactor*scaler.scale_ ] = np.nan
    xtrain3[ xtrain3 < scaler.center_ - stdFactor*scaler.scale_ ] = np.nan

    if xtrain3.isnan().any():
        print("NaN values assigned")

    sys.exit()

    xtest3[ xtest3 > scaler.center_ + stdFactor*scaler.scale_ ] = np.nan
    xtest3[ xtest3 < scaler.center_ - stdFactor*scaler.scale_ ] = np.nan

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(xtrain3)

    xtrain3.loc[:,:] = imp.transform(xtrain3)
    xtest3.loc[:,:]  = imp.transform(xtest3)

    # normalize features
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

    print()
    print("Using ElasticNet to determine features:")
    netCV = linear_model.ElasticNetCV(l1_ratio=[0., 0.1, 0.5, 0.75, 0.85, 0.9, 0.95, 1.],
    cv=5, n_jobs=2, max_iter=5e3, alphas=(0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.5))
    netCV.fit(xtrain4, ytrain.values.ravel())

    print("Selected nr of features: ", np.count_nonzero( netCV.coef_))
    print("Selected alpha: ", netCV.alpha_)
    print("Selected l1_ratio: ", netCV.l1_ratio_)
    net = linear_model.ElasticNet(l1_ratio=netCV.l1_ratio_, max_iter=5e3)
    print("Testing ElasticNet:")
    score = cross_val_score(net, xtrain4, ytrain.values.ravel(), cv=5, scoring='r2')
    print("score: ", score)
    net.fit(xtrain4, ytrain.values.ravel())

    xtrain5 = xtrain4.loc[:, net.coef_ > 0 ]
    xtest5 = xtest4.loc[:, net.coef_ > 0 ]

    print()
    print("Testing various regressors....")

    print("Testing ridge regression:")
    ridgeCV = linear_model.RidgeCV(alphas=np.linspace(0,3, 31).tolist(),
    cv=5, scoring='r2')
    ridgeCV.fit(xtrain5, ytrain.values.ravel())

    print("Selected alpha: ", ridgeCV.alpha_)

    ridge = linear_model.Ridge(alpha=ridgeCV.alpha_)
    score = cross_val_score(ridge, xtrain5, ytrain.values.ravel(), cv=5, scoring='r2')
    print("score: ", score)

    print("Testing RanodmForestRegressor:")
    rfr100 = ensemble.RandomForestRegressor(n_estimators=100, max_depth=15)
    score = cross_val_score(rfr100, xtrain5, ytrain.values.ravel(), cv=5, scoring='r2')
    print("For 100 Trees:")
    print("score: ", score)

    rfr500 = ensemble.RandomForestRegressor(n_estimators=500, max_depth=15)
    score = cross_val_score(rfr500, xtrain5, ytrain.values.ravel(), cv=5, scoring='r2')
    print("For 500 Trees:")
    print("score: ", score)

    rfr1000 = ensemble.RandomForestRegressor(n_estimators=1000, max_depth=15)
    score = cross_val_score(rfr1000, xtrain5, ytrain.values.ravel(), cv=5, scoring='r2')
    print("For 1000 Trees:")
    print("score: ", score)



if __name__ == "__main__":
    main()
    #KeepGaussianFeatures()


