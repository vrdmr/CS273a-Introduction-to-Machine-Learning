__author__ = 'varadmeru'

"train a regression MLP"

import numpy as np
import math
import cPickle as pickle
from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain.structure import TanhLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.structure import LinearLayer, SigmoidLayer, GaussianLayer
from pybrain.structure import BiasUnit
from pybrain.tools.validation import CrossValidator

import pandas as pd

from sklearn.metrics import mean_squared_error as MSE

split_at = 45000
train_file = '../data/kaggle/kaggle.X1.train.txt'
label_file = '../data/kaggle/kaggle.Y.train.txt'
validation_file = '../data/kaggle/kaggle.X1.test.txt'
model_file = 'model.pkl'
output_predictions_file = 'predictions.txt'


def train(train, label, custom_net=None, training_mse_threshold=0.40, testing_mse_threshold=0.60, epoch_threshold=10, epochs=100, hidden_size=20):
    # Test Set.
    x_train = train[0:split_at, :]
    y_train_slice = label.__getslice__(0, split_at)
    y_train = y_train_slice.reshape(-1, 1)
    x_test = train[split_at:, :]
    y_test_slice = label.__getslice__(split_at, label.shape[0])
    y_test = y_test_slice.reshape(-1, 1)

    # Shape.
    input_size = x_train.shape[1]
    target_size = y_train.shape[1]

    # prepare dataset
    ds = SDS(input_size, target_size)
    ds.setField('input', x_train)
    ds.setField('target', y_train)

    # prepare dataset
    ds_test = SDS(input_size, target_size)
    ds_test.setField('input', x_test)
    ds_test.setField('target', y_test)

    min_mse = 1000000

    # init and train
    if custom_net == None:
        net = buildNetwork(input_size, hidden_size, target_size, bias=True)
    else:
        print "Picking up the custom network"
        net = custom_net

    trainer = RPropMinusTrainer(net, dataset=ds, verbose=False, weightdecay=0.01, batchlearning=True)
    print "training for {} epochs...".format(epochs)

    for i in range(epochs):
        mse = trainer.train()
        print "training mse, epoch {}: {}".format(i + 1, math.sqrt(mse))

        p = net.activateOnDataset(ds_test)
        mse = math.sqrt(MSE(y_test, p))
        print "-- testing mse, epoch {}: {}".format(i + 1, mse)
        pickle.dump(net, open("current_run", 'wb'))

        if min_mse > mse:
            print "Current minimum found at ", i
            pickle.dump(net, open("current_min_epoch_" + model_file, 'wb'))
            min_mse = mse

    pickle.dump(net, open(model_file, 'wb'))
    return net


def train_cross_validate(train, label, custom_net=None, training_mse_threshold=0.40, testing_mse_threshold=0.60,
                         epoch_threshold=10, epochs=100, hidden_size=50):
    # Test Set.
    x_train = train[0:split_at, :]
    y_train_slice = label.__getslice__(0, split_at)
    y_train = y_train_slice.reshape(-1, 1)
    x_test = train[split_at:, :]
    y_test_slice = label.__getslice__(split_at, label.shape[0])
    y_test = y_test_slice.reshape(-1, 1)

    # Shape.
    input_size = x_train.shape[1]
    target_size = y_train.shape[1]

    input_size_test = x_test.shape[1]
    target_size_test = y_test.shape[1]

    # prepare dataset
    ds = SDS(input_size, target_size)
    ds.setField('input', x_train)
    ds.setField('target', y_train)

    # prepare dataset
    ds_test = SDS(input_size, target_size)
    ds_test.setField('input', x_test)
    ds_test.setField('target', y_test)

    min_mse = 1000000

    # init and train
    if custom_net == None:
        net = buildNetwork(input_size, hidden_size, target_size, bias=True, hiddenclass=TanhLayer)
    else:
        print "Picking up the custom network"
        net = custom_net

    trainer = RPropMinusTrainer(net, dataset=ds, verbose=True, weightdecay=0.01, batchlearning=True)
    print "training for {} epochs...".format(epochs)

    for i in range(epochs):
        mse = trainer.train()
        print "training mse, epoch {}: {}".format(i + 1, mse)

        p = net.activateOnDataset(ds_test)
        mse = MSE(y_test, p)
        print "-- testing mse, epoch {}: {}".format(i + 1, mse)
        pickle.dump(net, open("current_run", 'wb'))

        if min_mse > mse:
            print "Current minimum found at ", i
            pickle.dump(net, open("current_min_epoch_" + model_file, 'wb'))
            min_mse = mse

    pickle.dump(net, open(model_file, 'wb'))
    return net


def validate(X, y, net):
    # Test Set.
    x_test = X[split_at:, :]
    y_test = y.__getslice__(split_at, y.shape[0])
    y_test = y_test.reshape(-1, 1)

    # you'll need labels. In case you don't have them...
    y_test_dummy = np.zeros(y_test.shape)

    input_size = x_test.shape[1]
    target_size = y_test.shape[1]

    assert (net.indim == input_size)
    assert (net.outdim == target_size)

    # prepare dataset
    ds = SDS(input_size, target_size)
    ds.setField('input', x_test)
    ds.setField('target', y_test)

    # predict

    p = net.activateOnDataset(ds)

    mse = MSE(y_test, p)
    print "testing MSE:", mse
    np.savetxt(output_predictions_file, p, fmt='%.6f')


def predict(X, net):
    # Test Set.
    x_test = X[:, :]

    # you'll need labels. In case you don't have them...
    y_test_dummy = np.zeros((X.shape[0], 1))

    input_size = x_test.shape[1]
    target_size = y_test_dummy.shape[1]

    assert (net.indim == input_size)
    assert (net.outdim == target_size)

    # prepare dataset
    ds = SDS(input_size, target_size)
    ds.setField('input', x_test)
    ds.setField('target', y_test_dummy)

    p = net.activateOnDataset(ds)
    print p.shape
    np.savetxt("1_" + output_predictions_file, p, fmt='%.6f')
    s = pd.Series(p[:, 0])
    s.index += 1
    s.to_csv('neural_prediction_3.csv', header=['Prediction'], index=True, index_label='ID')


def build_deep_network(linear_dimensions):
    neural_net = FeedForwardNetwork()

    inLayer = LinearLayer(linear_dimensions)
    hiddenLayer_1 = SigmoidLayer(100)
    hiddenLayer_2 = SigmoidLayer(100)
    hiddenLayer_3 = SigmoidLayer(50)
    outLayer = LinearLayer(1)

    neural_net.addInputModule(inLayer)
    neural_net.addModule(hiddenLayer_1)
    neural_net.addModule(hiddenLayer_2)
    neural_net.addModule(hiddenLayer_3)
    neural_net.addOutputModule(outLayer)

    in_to_hidden_1 = FullConnection(inLayer, hiddenLayer_1)
    hidden_1_to_hidden_2 = FullConnection(hiddenLayer_1, hiddenLayer_2)
    hidden_2_to_hidden_3 = FullConnection(hiddenLayer_2, hiddenLayer_3)
    hidden_3_to_output = FullConnection(hiddenLayer_3, outLayer)

    neural_net.addConnection(in_to_hidden_1)
    neural_net.addConnection(hidden_1_to_hidden_2)
    neural_net.addConnection(hidden_2_to_hidden_3)
    neural_net.addConnection(hidden_3_to_output)

    neural_net.sortModules()
    return neural_net

# load data
train_data = np.loadtxt(train_file, delimiter=',')
# train_data_2 = train_data1 - np.mean(train_data1, axis=0)
# trmax, trmin = train_data_2.max(axis=0), train_data_2.min(axis=0)
# train_data = (train_data_2 - trmin) / (trmax - trmin)

validation_data = np.loadtxt(validation_file, delimiter=',')
# validation_data2 = validation_data1 - np.mean(validation_data1, axis=0)
# vmax, vmin = validation_data2.max(), validation_data2.min()
# validation_data = (validation_data2 - vmin) / (vmax - vmin)

labels = np.loadtxt(label_file, delimiter=',')
threshold = 0.34

neural = build_deep_network(train_data.shape[1])
learner_net = train(train_data, labels, custom_net=neural, training_mse_threshold=threshold, epoch_threshold=20,
                    epochs=50)

validation_net = pickle.load(open("current_min_epoch_model.pkl", 'rb'))
predict(validation_data, validation_net)

# net_load = pickle.load(open(model_file, 'rb'))
# validate(train_data, labels, net_load)