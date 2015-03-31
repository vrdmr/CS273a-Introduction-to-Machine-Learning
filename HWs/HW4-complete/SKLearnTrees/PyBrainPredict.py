__author__ = 'varadmeru'

"get predictions for a test set"

import numpy as np
import cPickle as pickle

from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from sklearn.metrics import mean_squared_error as MSE

test_file  = '../data/kaggle/kaggle.X1.test.txt'
model_file = 'model.pkl'
output_predictions_file = 'predictions.txt'

# load model

net = pickle.load(open(model_file, 'rb'))

# load data

test = np.loadtxt(test_file, delimiter=',')

x_test = test[:, 0:-1]
y_test = test[:, -1]
y_test = y_test.reshape(-1, 1)

# you'll need labels. In case you don't have them...
y_test_dummy = np.zeros(y_test.shape)

input_size = x_test.shape[1]
target_size = y_test.shape[1]

assert ( net.indim == input_size )
assert ( net.outdim == target_size )

# prepare dataset

ds = SDS(input_size, target_size)
ds.setField('input', x_test)
ds.setField('target', y_test_dummy)

# predict

p = net.activateOnDataset(ds)

mse = MSE(y_test, p)
rmse = sqrt(mse)

print "testing RMSE:", rmse

np.savetxt(output_predictions_file, p, fmt='%.6f')