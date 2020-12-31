import pandas as pd
import numpy as np
from scipy.stats import skew
from numpy.linalg import inv
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())
#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice


def normal_linear(x_train, t):
    x_train = np.array(x_train)
    p = np.transpose(x_train) @ x_train
    t = np.array(t)
    t = np.reshape(t, (np.shape(t)[0], 1))
    te = np.transpose(x_train) @ t
    return inv(p)@te


def ridge_linear(x_train, t, lam):
    x_train = np.array(x_train)
    p = np.transpose(x_train) @ x_train
    l = lam * np.identity(np.shape(p)[0])
    p_rid = p+l
    t = np.array(t)
    t = np.reshape(t, (np.shape(t)[0], 1))
    te = np.transpose(x_train) @ t
    return inv(p_rid) @ te

w_ridge = ridge_linear(X_train, y, lam=1)
w_norm = normal_linear(X_train, y)
X_test = np.array(X_test)
y_test_norm = []
y_test_ridge = []
for i in X_test:
    a = w_norm.T @ i
    b = w_ridge.T @ i
    y_test_norm.append(int(a))
    y_test_ridge.append(int(b))
print(y_test_norm)
print(y_test_ridge)
