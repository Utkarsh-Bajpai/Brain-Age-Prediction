import pandas as pd
import numpy as np
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

import numpy as np

def outliers_modified_z_score(ys):
    threshold = 3.5

    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y for y in ys]
    return np.where(np.abs(modified_z_scores) > threshold)

def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))

data_path=""

X_test=pd.read_csv(data_path+"X_test.csv",header=0)
X_train=pd.read_csv(data_path+"X_train.csv",header=0)
Y_train=pd.read_csv(data_path+"Y_train.csv",header=0)

X_train.drop(['id'], axis=1)

#cleaning dataset
for column in X_train.columns.values:
    replacement = X_train[column].median()
    X_train[column].fillna(replacement, inplace=True)
    X_train[column].replace("", replacement, inplace=True)
    X_train[column].replace(" ", replacement, inplace=True)

for column in X_test.columns.values:
    replacement = X_test[column].median()
    X_test[column].fillna(replacement, inplace=True)
    X_test[column].replace("", replacement, inplace=True)
    X_test[column].replace(" ", replacement, inplace=True)


#Outlier Detection
for column in X_train.columns.values:
    outliers = outliers_iqr(X_train[column].values)
    if(len(outliers[0])>0):
        t = X_train[column].values
        t = np.delete(t, outliers[0])
        for outlier in outliers[0]:
            X_train.set_value(outlier, column, np.median(t))

for column in X_train.columns.values:
    outliers = outliers_modified_z_score(X_train[column].values)
    if(len(outliers[0])>0):
        t = X_train[column].values
        t = np.delete(t, outliers[0])
        for outlier in outliers[0]:
            X_train.set_value(outlier, column, np.median(t))

#X_train.set_index('id').to_csv(data_path+"median_modified_z_score.csv")

X_train.set_index('id').to_csv(data_path+"median_modified_iqr_zscore.csv")


#feature selection
# Create correlation matrix
corr_matrix = X_train.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
print(to_drop)

# Drop features 
X_train = X_train.drop(to_drop, 1)

X_train.set_index('id').to_csv(data_path+"Final Dataset.csv")


#Drop Features with Low variance
selector = VarianceThreshold()
X_train = selector.fit_transform(X_train)

print(X_train.shape)

lm = linear_model.LinearRegression()
model = lm.fit(X_train,Y_train)

y_pred=lm.predict(X_test)

#score = r2_score(y, y_pred)

#test["y"]=test.loc[:,"x1":"x10"].mean(axis=1)

#test[["Id","y"]].set_index("Id").to_csv(data_path+"res.csv")
