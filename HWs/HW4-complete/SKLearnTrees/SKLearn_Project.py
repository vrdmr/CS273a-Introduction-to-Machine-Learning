__author__ = 'varadmeru'

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing.data import normalize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt

from sklearn import linear_model, svm, datasets, feature_selection, cross_validation, preprocessing


X = pd.read_csv('../data/kaggle/kaggle.X1.train.txt', header=None)
Y = pd.read_csv('../data/kaggle/kaggle.Y.train.txt', header=None)
Xtest = pd.read_csv('../data/kaggle/kaggle.X1.test.txt', header=None)
Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.25, random_state=42)

print Xtr.shape
print Xte.shape
print Ytr.shape
print Yte.shape
print "****************************************"


def apply_svm_rbf(gamma=0.1, c=1e3):
    svr_rbf = SVR(kernel='rbf', C=c, gamma=gamma, verbose=True)
    y_rbf = svr_rbf.fit(Xtr, Ytr[:, 0]).predict(X)
    mse_rbf = mean_squared_error(Yte, y_rbf)
    return y_rbf, mse_rbf


def apply_svm_linear(c=1e3):
    svr_lin = SVR(kernel='linear', C=c, verbose=True)
    y_lin = svr_lin.fit(Xtr, Ytr[:, 0]).predict(X)
    mse_lin = mean_squared_error(Yte, y_lin)
    return y_lin, mse_lin


def apply_svm_poly(degree=2, c=1e3):
    svr_poly = SVR(kernel='poly', C=c, degree=degree, verbose=True)
    y_poly = svr_poly.fit(Xtr, Ytr[:, 0]).predict(X)
    mse_poly = mean_squared_error(Yte, y_poly)
    return y_poly, mse_poly


def apply_knn():
    regr = KNeighborsRegressor()
    regr.fit(Xtr, Ytr)

    pred = regr.predict(Xte)
    temp = mean_squared_error(Yte, pred)
    return pred, temp


def apply_linear_regression():
    regr = linear_model.LinearRegression(normalize=True)
    regr.fit(Xtr, Ytr)

    pred = regr.predict(Xte)
    temp = mean_squared_error(Yte, pred)
    return pred, temp, regr.coef_


def write_predictions_to_file(pred):
    s = pd.Series(pred)
    s.index = s.index + 1
    s.to_csv('pyprediction.csv', header=['Prediction'], index=True, index_label='ID')


yhat, mse, coef = apply_linear_regression()
print "apply_linear_regression", mse

yhat, mse = apply_knn()
print "apply_knn", mse

yhat, mse = apply_svm_poly()
print "apply_svm", mse

yhat, mse = apply_svm_rbf()
print "apply_svm", mse

yhat, mse = apply_svm_linear()
print "apply_svm", mse

# write_predictions_to_file(p)

'''

# Generate sample data
X = np.sort(5 * np.random.rand(40, 10), axis=0)
print X.shape[0]
y = np.sin(X[:,0])
print y.shape

###############################################################################
# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

###############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)

y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

print svr_rbf
print svr_lin
print svr_poly


###############################################################################
# look at the results
plt.scatter(X, y, c='k', label='data')
plt.hold('on')
plt.plot(X, y_rbf, c='g', label='RBF model')
plt.plot(X, y_lin, c='r', label='Linear model')
plt.plot(X, y_poly, c='b', label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
'''
