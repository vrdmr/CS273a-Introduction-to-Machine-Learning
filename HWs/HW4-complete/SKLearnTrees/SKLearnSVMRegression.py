__author__ = 'varadmeru'

from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import *
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import math
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


def svr1():
    boston = load_boston()
    print boston.data.shape

    X = pd.read_csv('../data/kaggle/kaggle.X1.train.txt', header=None)
    Y = pd.read_csv('../data/kaggle/kaggle.Y.train.txt', header=None)
    Xtest = pd.read_csv('../data/kaggle/kaggle.X1.test.txt', header=None)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    print X_train.shape
    print X_test.shape
    print y_train.shape
    print y_test.shape

    scalerX = StandardScaler().fit(X_train)
    scalery = StandardScaler().fit(y_train)
    X_train = scalerX.transform(X_train)
    y_train = scalery.transform(y_train)

    X_test = scalerX.transform(X_test)
    y_test = scalery.transform(y_test)

    model = LinearRegression(normalize=True)
    train_and_evaluate(model, X_train, y_train, X_test, y_test)

    model = LinearRegression()
    train_and_evaluate(model, X_train, y_train, X_test, y_test)

    model = linear_model.BayesianRidge()
    train_and_evaluate(model, X_train, y_train, X_test, y_test)

    for degree in [3, 4, 5]:
        print "For :", degree
        model = make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)
        print "Validation RMSE", math.sqrt(mean_squared_error(y_test, yhat))


def train_and_evaluate(clf, X_train, y_train_, X_test, y_test):
    print "===================================================================="
    print "In T and E, with clf -", clf
    y_train = np.ravel(y_train_)
    clf.fit(X_train, y_train)

    yhat = clf.predict(X_test)
    print "Validation RMSE", math.sqrt(mean_squared_error(y_test, yhat))
    print "-----------------------------------------------------------------"


def process():


    Y = pd.read_csv('../data/kaggle/kaggle.Y.train.txt', header=None)
    Xtest = pd.read_csv('../data/kaggle/kaggle.X1.test.txt', header=None)
    Xnames1 = pd.read_csv('../data/kaggle/kaggle.X1.names.txt', header=None)
    Xnames = pd.DataFrame(Xnames1)[:]
    X = pd.read_csv('../data/kaggle/kaggle.X1.train.txt', header=None)
    X.header = Xnames[:]
    X["label"] = Y

    X.to_csv('kaggle.X1.train.with.labels.csv', header=True, index=False)


process()
# example()

'''
print __doc__

###############################################################################
# Generate sample data
import numpy as np

X = np.sort(5 * np.random.rand(40, 2), axis=0)
print X.size
y = np.sin(X[:,0]).ravel()
print y.size

###############################################################################
# Add noise to targets
#y[::5] += 3 * (0.5 - np.random.rand(8))

###############################################################################
# Fit regression model
from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

###############################################################################
# look at the results
import pylab as pl
pl.scatter(X, y, c='k', label='data')
pl.hold('on')
pl.plot(X, y_rbf, c='g', label='RBF model')
pl.plot(X, y_lin, c='r', label='Linear model')
pl.plot(X, y_poly, c='b', label='Polynomial model')
pl.xlabel('data')
pl.ylabel('target')
pl.title('Support Vector Regression')
pl.legend()
pl.show()
'''