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
"""
For Hyperbolic Tangents..

/usr/local/bin/python /Users/varadmeru/uci-related/uci-courses/CS273a-Introduction-to-Machine-Learning/HWs/HW4-complete/SKLearnTrees/PyBrainImplementation.py
Picking up the custom network
training for 100 epochs...
training mse, epoch 1: 0.44356518946
training mse, epoch 2: 0.440433224972
training mse, epoch 3: 0.437699969101
training mse, epoch 4: 0.436009602511
training mse, epoch 5: 0.435051688376
training mse, epoch 6: 0.433664385232
"""

"""
inLayer = LinearLayer(linear_dimensions)
hiddenLayer_1 = SigmoidLayer(100)
hiddenLayer_2 = SigmoidLayer(100)
hiddenLayer_3 = SigmoidLayer(50)
outLayer = LinearLayer(1)

/usr/local/bin/python /Users/varadmeru/uci-related/uci-courses/CS273a-Introduction-to-Machine-Learning/HWs/HW4-complete/SKLearnTrees/PyBrainImplementation.py
Picking up the custom network
training for 100 epochs...
training mse, epoch 1: 0.385418006979
training mse, epoch 2: 0.376363933568
training mse, epoch 3: 0.373001579916
training mse, epoch 4: 0.37170327199
training mse, epoch 5: 0.369864562523
training mse, epoch 6: 0.368622141371
training mse, epoch 7: 0.367093180267
training mse, epoch 8: 0.365900181508
training mse, epoch 9: 0.362574727975
training mse, epoch 10: 0.361591888905
...
training mse, epoch 26: 0.362082972512
training mse, epoch 27: 0.362219784098
training mse, epoch 28: 0.361474369181
training mse, epoch 29: 0.361831391946
training mse, epoch 30: 0.361465685255
training mse, epoch 31: 0.362298672028
training mse, epoch 32: 0.361799156738
training mse, epoch 33: 0.361180822353
training mse, epoch 34: 0.361870538739
training mse, epoch 35: 0.361695335471
training mse, epoch 36: 0.362156838895
"""

"""
inLayer = LinearLayer(linear_dimensions)
hiddenLayer_1 = TanhLayer(100)
hiddenLayer_2 = SigmoidLayer(50)
hiddenLayer_3 = SigmoidLayer(20)
outLayer = LinearLayer(1)

Picking up the custom network
training for 100 epochs...
training mse, epoch 1: 0.329669713737
training mse, epoch 2: 0.357733300374
training mse, epoch 3: 0.35685779179
training mse, epoch 4: 0.355170899093
training mse, epoch 5: 0.353106573759
training mse, epoch 6: 0.35264020698
training mse, epoch 7: 0.352327375314
training mse, epoch 8: 0.352553877003
training mse, epoch 9: 0.352401097694
training mse, epoch 10: 0.35241734765
training mse, epoch 11: 0.352397229931
training mse, epoch 12: 0.352359687132
training mse, epoch 13: 0.350472275736
training mse, epoch 14: 0.350678570722
training mse, epoch 15: 0.350800259934
training mse, epoch 16: 0.350800367883
training mse, epoch 17: 0.35040392669
training mse, epoch 18: 0.350297505079
training mse, epoch 19: 0.350489065891
training mse, epoch 20: 0.350567484196
"""

"""
inLayer = LinearLayer(linear_dimensions)
hiddenLayer_1 = SigmoidLayer(100)
hiddenLayer_2 = LinearLayer(10)
hiddenLayer_3 = SigmoidLayer(30)
outLayer = LinearLayer(1)

/usr/local/bin/python /Users/varadmeru/uci-related/uci-courses/CS273a-Introduction-to-Machine-Learning/HWs/HW4-complete/SKLearnTrees/PyBrainImplementation.py
No fast networks available.
Picking up the custom network
training for 100 epochs...
training mse, epoch 1: 0.359706775065
training mse, epoch 2: 0.364085977574
training mse, epoch 3: 0.363961003472
training mse, epoch 4: 0.365089280327
training mse, epoch 5: 0.364476257896
training mse, epoch 6: 0.364219112978
training mse, epoch 7: 0.365213940278
training mse, epoch 8: 0.364245748179
training mse, epoch 9: 0.364530269518
training mse, epoch 10: 0.364364129227
training mse, epoch 11: 0.364251393911
training mse, epoch 12: 0.363889965027



inLayer = LinearLayer(linear_dimensions)
hiddenLayer_1 = SigmoidLayer(50)
hiddenLayer_2 = LinearLayer(25)
hiddenLayer_3 = SigmoidLayer(10)
outLayer = LinearLayer(1)

/usr/local/bin/python /Users/varadmeru/uci-related/uci-courses/CS273a-Introduction-to-Machine-Learning/HWs/HW4-complete/SKLearnTrees/PyBrainImplementation.py
No fast networks available.
Picking up the custom network
training for 100 epochs...
training mse, epoch 1: 0.348505961542
training mse, epoch 2: 0.34079557954
training mse, epoch 3: 0.348752829307
training mse, epoch 4: 0.348849763617
training mse, epoch 5: 0.348450096177
training mse, epoch 6: 0.348495340788



-------------------------------
Simple Build Network, without our deep network

/usr/local/bin/python /Users/varadmeru/uci-related/uci-courses/CS273a-Introduction-to-Machine-Learning/HWs/HW4-complete/SKLearnTrees/PyBrainImplementation.py
No fast networks available.
training for 10 epochs...
training mse, epoch 1: 0.473880150337
-- testing mse, epoch 1: 0.96097563728
training mse, epoch 2: 0.461280156687
-- testing mse, epoch 2: 0.807230547454
training mse, epoch 3: 0.461878553865
-- testing mse, epoch 3: 0.894349179792
training mse, epoch 4: 0.461130559646
-- testing mse, epoch 4: 0.930172696681
training mse, epoch 5: 0.460438101055
-- testing mse, epoch 5: 0.763341293704
training mse, epoch 6: 0.462587026178
-- testing mse, epoch 6: 0.727991003352
training mse, epoch 7: 0.458525505935
-- testing mse, epoch 7: 0.836220586735
training mse, epoch 8: 0.460225395198
-- testing mse, epoch 8: 0.79176639014
training mse, epoch 9: 0.460772594876
-- testing mse, epoch 9: 1.12788450007
training mse, epoch 10: 0.461544952522
-- testing mse, epoch 10: 1.77699234491

Process finished with exit code 0

"""

"""
inLayer = LinearLayer(linear_dimensions)
    hiddenLayer_1 = SigmoidLayer(50)
    biasUnit = BiasUnit(name='bias')
    hiddenLayer_2 = TanhLayer(25)
    hiddenLayer_3 = SigmoidLayer(10)
    outLayer = LinearLayer(1)

Your submission scored 0.88306, which is not an improvement of your best score. Keep trying!
Your submission scored 0.83643, which is not an improvement of your best score. Keep trying!
"""

"""
/usr/local/bin/python /Users/varadmeru/uci-related/uci-courses/CS273a-Introduction-to-Machine-Learning/HWs/HW4-complete/SKLearnTrees/PyBrainImplementation.py
(60000, 91)
(40000, 91)
Picking up the custom network
training for 100 epochs...
training mse, epoch 1: 0.457129010867
-- testing mse, epoch 1: 122.184860211
epoch      1  total error       61.217   avg weight       0.99596
training mse, epoch 2: 61.2172583086
-- testing mse, epoch 2: 0.951942237007
epoch      2  total error      0.45629   avg weight       0.99313
training mse, epoch 3: 0.456292824589
-- testing mse, epoch 3: 31.0644933357
epoch      3  total error       15.586   avg weight       0.99668
training mse, epoch 4: 15.5855576275
-- testing mse, epoch 4: 0.908546406761
epoch      4  total error      0.43497   avg weight       0.99457
training mse, epoch 5: 0.434973383353
-- testing mse, epoch 5: 7.43081516392
epoch      5  total error       3.7325   avg weight       0.99615
training mse, epoch 6: 3.73245877379
-- testing mse, epoch 6: 0.895899072599
epoch      6  total error      0.42883   avg weight       0.99211
training mse, epoch 7: 0.4288319691
-- testing mse, epoch 7: 1.95272157884
epoch      7  total error      0.97535   avg weight       0.99234
training mse, epoch 8: 0.975352059856
-- testing mse, epoch 8: 0.89199008836
epoch      8  total error      0.42697   avg weight       0.99056
training mse, epoch 9: 0.42696510959
-- testing mse, epoch 9: 0.863500523387
epoch      9  total error      0.42176   avg weight       0.99035
training mse, epoch 10: 0.421755004882
-- testing mse, epoch 10: 0.890637158239
epoch     10  total error      0.42633   avg weight       0.98915
training mse, epoch 11: 0.426331587241
-- testing mse, epoch 11: 0.736605039407
epoch     11  total error      0.35383   avg weight       0.98907
training mse, epoch 12: 0.353827611508
-- testing mse, epoch 12: 0.731794095337
epoch     12  total error      0.35067   avg weight       0.98803
training mse, epoch 13: 0.350671531774
-- testing mse, epoch 13: 0.760074002045
epoch     13  total error      0.36835   avg weight       0.98725
training mse, epoch 14: 0.368351527145
-- testing mse, epoch 14: 0.739608198677
epoch     14  total error      0.35408   avg weight       0.98655
training mse, epoch 15: 0.354082627347
-- testing mse, epoch 15: 0.719543266433
epoch     15  total error      0.34603   avg weight         0.986
training mse, epoch 16: 0.346026319767
-- testing mse, epoch 16: 0.738783870854
epoch     16  total error       0.3536   avg weight       0.98533
training mse, epoch 17: 0.353604493971
-- testing mse, epoch 17: 0.720574555947
epoch     17  total error      0.34547   avg weight        0.9861
training mse, epoch 18: 0.3454667746
-- testing mse, epoch 18: 0.714528750337
epoch     18  total error       0.3434   avg weight       0.98569
training mse, epoch 19: 0.343398994959
-- testing mse, epoch 19: 0.723169823536
epoch     19  total error      0.34645   avg weight       0.98533
training mse, epoch 20: 0.346451499506
-- testing mse, epoch 20: 0.71636968954
epoch     20  total error      0.34373   avg weight       0.98526
training mse, epoch 21: 0.343728025349
-- testing mse, epoch 21: 0.713982018717
epoch     21  total error       0.3431   avg weight       0.98492
training mse, epoch 22: 0.343100561358
-- testing mse, epoch 22: 0.716919280645
epoch     22  total error      0.34381   avg weight       0.98456
training mse, epoch 23: 0.343808364673
-- testing mse, epoch 23: 0.714667974725
epoch     23  total error      0.34309   avg weight       0.98468
training mse, epoch 24: 0.343088721751
-- testing mse, epoch 24: 0.713706599463
epoch     24  total error      0.34294   avg weight       0.98435
training mse, epoch 25: 0.342943650295
-- testing mse, epoch 25: 0.714685949644
epoch     25  total error      0.34298   avg weight       0.98403
training mse, epoch 26: 0.342978165609
-- testing mse, epoch 26: 0.713644178304
epoch     26  total error      0.34265   avg weight       0.98374
training mse, epoch 27: 0.342647936116
-- testing mse, epoch 27: 0.71317704983
epoch     27  total error       0.3426   avg weight       0.98339
training mse, epoch 28: 0.342598802811
-- testing mse, epoch 28: 0.713793362819
epoch     28  total error      0.34268   avg weight       0.98303
training mse, epoch 29: 0.342682335392
-- testing mse, epoch 29: 0.713379389552
epoch     29  total error      0.34259   avg weight       0.98338
training mse, epoch 30: 0.342586466378
-- testing mse, epoch 30: 0.713144449967
epoch     30  total error      0.34259   avg weight       0.98312
training mse, epoch 31: 0.342593605546
-- testing mse, epoch 31: 0.713513689145
epoch     31  total error      0.34261   avg weight        0.9829
training mse, epoch 32: 0.342608003758
-- testing mse, epoch 32: 0.71325743671
epoch     32  total error      0.34258   avg weight       0.98311
training mse, epoch 33: 0.342576782259
-- testing mse, epoch 33: 0.713308958431
epoch     33  total error      0.34257   avg weight       0.98296
training mse, epoch 34: 0.342568559889
-- testing mse, epoch 34: 0.713185530838
epoch     34  total error      0.34257   avg weight       0.98283
training mse, epoch 35: 0.342566323492
-- testing mse, epoch 35: 0.713292135298
epoch     35  total error      0.34256   avg weight       0.98269
training mse, epoch 36: 0.342562010405
-- testing mse, epoch 36: 0.713218754666
epoch     36  total error      0.34255   avg weight       0.98258
training mse, epoch 37: 0.342552351536
-- testing mse, epoch 37: 0.713156919749
epoch     37  total error      0.34255   avg weight       0.98246
training mse, epoch 38: 0.342546179104
-- testing mse, epoch 38: 0.713191357906
epoch     38  total error      0.34254   avg weight       0.98238
training mse, epoch 39: 0.342538495455
-- testing mse, epoch 39: 0.713131111817
epoch     39  total error      0.34253   avg weight       0.98237
training mse, epoch 40: 0.342529268601
-- testing mse, epoch 40: 0.71312700482
"""