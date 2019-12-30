#! /bin/python3

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import csv

from sklearn.impute import SimpleImputer
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn import linear_model
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

computeDistances = False



# fill in missing data
def ImputeMean(data):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    data.iloc[:,:] = imp.fit_transform(data)
    return data

def ImputeMedian(data):
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    data.iloc[:,:] = imp.fit_transform(data)
    return data

def ComputeMahalanobisDistance(data):
    """Compute MahalanobisDistance and return as DataFrame

    Parameters:
    data (DataFrame): Pandas DataFrame
    
    Returns:
    DataFrame: contains mahalanobis distances with indices from data
    """
    rob_cov = MinCovDet().fit(data)
    distances = rob_cov.mahalanobis(data)
    distances = np.sqrt(distances)

    df = pd.DataFrame(
        data=distances,
        columns={'distance'},
        index=data.index.values
    )
    return df



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

    xtrain = ImputeMean(xtrain)
    xtest = ImputeMean(xtest)

    # drop columns with low variance
    varThre = VarianceThreshold(threshold=1e-4)
    varThre.fit(xtrain)
    print("Will drop ", xtrain.columns[ np.invert(varThre.get_support())])
    xtrain = pd.DataFrame(
        data=varThre.transform(xtrain),
        index=xtrain.index.tolist(),
        columns= xtrain.columns[ varThre.get_support()].tolist()
    )
    xtest = pd.DataFrame(
        data=varThre.transform(xtest),
        index=xtest.index.tolist(),
        columns=xtest.columns[ varThre.get_support()].tolist()
    )

    if(computeDistances):
        print("Computing Distances")
        distance_df = ComputeMahalanobisDistance(xtrain)
        distance_df.to_csv('distances.csv')

        #plot distances histogram
        plt.hist(distance_df, bins=200)
        plt.title('Mahalanobis Distance')
        plt.savefig('distances.png')
    else:
        print("Reading Distances from Disk")
        distance_df = pd.read_csv('distances.csv', index_col=0)

    # remove outliers
    xtrain = xtrain[ distance_df['distance'] <= 8.0]
    ytrain = ytrain[ distance_df['distance'] <= 8.0]

    # normalize data
    print("Normalizing Data")
    scaler = StandardScaler()
    scaler.fit(xtrain)

    xnormalized = pd.DataFrame(
        data=scaler.transform( xtrain),
        index=xtrain.index.tolist(),
        columns=xtrain.columns.tolist()
    )
    ynormalized = ytrain

    # normalize test data with same scaler
    xnormalized_test= pd.DataFrame(
        data=scaler.transform( xtest ),
        index=xtest.index.tolist(),
        columns=xtest.columns.tolist()
    )

    # perform feature selection via Recursive Feature Elimination
    model1 = linear_model.LinearRegression()
    rfecv = RFECV(
        estimator=model1,
        step=1,
        min_features_to_select=180,
        cv=KFold(5),
        scoring='r2',
        verbose=1
    )

    #rfecv.fit(xnormalized, ynormalized.values.ravel())

    # select features and train model
    #model = linear_model.LinearRegression()

    #xrelevant = xnormalized[ xnormalized.columns[rfecv.get_support()]]
    #xrelevant_test = xnormalized_test[ xnormalized_test.columns[rfecv.get_support()]]

    #model.fit(xrelevant, ynormalized.values)

    xrelevant = xnormalized
    xrelevant_test = xnormalized_test

    # model = linear_model.Lasso(
    #     alpha=0.1,
    #     fit_intercept=False
    # )

    model = linear_model.LinearRegression(fit_intercept=False)

    # model = linear_model.LassoCV(
    #     cv=5, 
    #     fit_intercept=False, 
    #     verbose=1,
    #     n_jobs=3)
    model.fit(xrelevant, ynormalized.values.ravel())
    print("Score: ", model.score(xrelevant, ynormalized))
    ypred = model.predict(xrelevant_test)

    y = pd.DataFrame(
        data=ypred,
        index=xrelevant_test.index.tolist(),
        columns={'y'}
    )
    y.index.rename('id', inplace=True)
    y.index = y.index.astype(np.float64)
    y.to_csv('prediction_lasso.csv')


def main_simpler():
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
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(xtrain)

    xtrain.loc[:,:] = imp.transform(xtrain.values)
    xtest.loc[:,:] = imp.transform(xtest.values)

    # drop features with no in-feature variance
    thresholder = VarianceThreshold(threshold=1e-8)
    thresholder.fit(xtrain)
    print("Will drop: ", xtrain.columns[ np.invert(thresholder.get_support())])
    
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

    # scale features via robust estimate
    #estimate = MinCovDet().fit(xtrain2)
    #covar = estimate.covariance_
    #mean = estimate.location_
    scaler = RobustScaler(quantile_range=(0.05, 0.95))
    scaler.fit(xtrain2)

    xtrain2.loc[:,:] = scaler.transform(xtrain2)
    xtest2.loc[:,:] = scaler.transform(xtest2)


    # data is now centered and has unit variance, ie 2/3 of data is
    # in unit ball

    # remove outliers -> upper/lower bound large deviations from mean
    xtrain2[ xtrain2 < -3.] = 0
    xtrain2[ xtrain2 >  3.] =  0

    xtest2[ xtest2 < -3.] = 0
    xtest2[ xtest2 >  3.] = 0

    # drop highly correlated features
    covar = xtrain2.cov()
    covar_triu = covar.where( np.triu(np.ones(covar.shape),k=1).astype(np.bool) )
    to_drop = [column for column in covar_triu.columns if any(covar_triu[column] > 0.95)]

    print("Will drop: ", to_drop)
    xtrain2 = xtrain2.drop(columns= to_drop)
    xtest2 = xtest2.drop(columns= to_drop)

    print("Remaining data: ", xtrain2.shape)

    #model = linear_model.LinearRegression()
    #rfecv = RFECV(
    #    estimator=model,
    #    step=10,
    #    min_features_to_select=180,
    #    cv=KFold(5),
    #    scoring='r2',
    #    verbose=0
    #)

    #rfecv.fit(xtrain2, ytrain.values.ravel())

    #print("Selected nr of features: ", rfecv.n_features_)

    #xtrain_rel = xtrain2[ xtrain2.columns[rfecv.get_support()]]
    #test_rel = xtest2[ xtest2.columns[rfecv.get_support()]


    #model2 = linear_model.LinearRegression()

    #score = cross_val_score(estimator=model2, X=xtrain_rel, y=ytrain.values.ravel(),
    #    scoring='r2', cv=5, n_jobs=3)

    #print("Score: ", score)

    model = linear_model.LassoCV(n_jobs=3, verbose=0, cv=5, max_iter=5e3)
    model.fit(xtrain2, ytrain.values.ravel())

    print("Selected nr of features: ", np.count_nonzero( model.coef_))
    print("Selected alpha: ", model.alpha_)
    model2 = linear_model.Lasso(
        alpha=model.alpha_
    )

    score = cross_val_score(estimator=model2, X=xtrain2, y=ytrain.values.ravel(),
        scoring='r2', cv=5, n_jobs=3)

    print("Score: ", score)

    #model2 = linear_model.LinearRegression()
    model2.fit(xtrain2, ytrain.values.ravel())
    pred = model2.predict(xtest2)

    pred_df = pd.DataFrame(
        data=pred,
        index=xtest2.index.tolist(),
        columns={'y'}
    )
    pred_df.index.rename('id', inplace=True)
    pred_df.index = pred_df.index.astype(np.float64)
    pred_df.to_csv('prediction_simpler.csv')





if __name__ == "__main__":
    computeDistances = False
    main_simpler()

