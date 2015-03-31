from sklearn import tree
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import math

import warnings


def main():
    X = pd.read_csv('../data/kaggle/kaggle.X1.train.txt', header=None)
    Y = pd.read_csv('../data/kaggle/kaggle.Y.train.txt', header=None)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.25, random_state=42)

    mxDepth = 20;
    smallestD = 0;
    smallestP = 0;
    smallestMSE = 10000000;

    for d in range(6, 13):
        print "Current Depth", d
        for p in range(1, 1000, 50):
            clf = tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=d, min_samples_split=p)
            clf = clf.fit(Xtr, Ytr)
            y_pred = clf.predict(Xte)
            temp = mean_squared_error(Yte, y_pred)
            if smallestMSE > temp:
                print "P|D|Temp - ", p, d, temp
                smallestP = p
                smallestD = d
                smallestMSE = temp

    print "P|D|Temp - ", smallestP, smallestD, smallestMSE


'''
/usr/local/bin/python /Users/varadmeru/uci-related/uci-courses/CS273a-Introduction-to-Machine-Learning/HWs/HW4-complete/SKLearnTrees/SKLearnTree.py
Current Depth 1
P|D|Temp -  1 1 0.574501067507
Current Depth 2
P|D|Temp -  1 2 0.52308805474
Current Depth 3
P|D|Temp -  1 3 0.486111169293
Current Depth 4
P|D|Temp -  1 4 0.467676157059
P|D|Temp -  8 4 0.467676157059
Current Depth 5
P|D|Temp -  1 5 0.454070418516
Current Depth 6
P|D|Temp -  1 6 0.443630009619
P|D|Temp -  2 6 0.443506369803
P|D|Temp -  4 6 0.443367099959
Current Depth 7
P|D|Temp -  1 7 0.437632047463
P|D|Temp -  4 7 0.437570934834
P|D|Temp -  5 7 0.437002045602
P|D|Temp -  7 7 0.436046059169
P|D|Temp -  8 7 0.435883885824
P|D|Temp -  9 7 0.435722604968
Current Depth 8
P|D|Temp -  8 8 0.434212563996
P|D|Temp -  9 8 0.432982583028
Current Depth 9
P|D|Temp -  9 9 0.431391825893
Current Depth 10
P|D|Temp -  9 10 0.428941612736
Current Depth 11
Current Depth 12
Current Depth 13
Current Depth 14
'''


def cv():
    X = pd.read_csv('../data/kaggle/kaggle.X1.train.txt', header=None)
    Y = pd.read_csv('../data/kaggle/kaggle.Y.train.txt', header=None)

    smallestD = 0;
    smallestP = 0;
    smallestMSE = 10000000;

    for d in range(8, 12):
        print "Current Depth", d
        for p in range(6, 12):
            print "Current MinParent", p
            clf = tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=d, min_samples_split=2 ** p)
            scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
            print scores
            temp = scores.mean()
            if smallestMSE > temp:
                print "P|D|Temp - ", p, d, temp
                smallestP = p
                smallestD = d
                smallestMSE = temp

    print "P|D|Temp - ", smallestP, smallestD, smallestMSE

    # clf = tree.DecisionTreeRegressor(criterion='mse', splitter='best',max_depth=9, min_samples_split=2**10)
    # scores = cross_validation.cross_val_score(clf, X, Y, cv=5)


def single():
    X = pd.read_csv('../data/kaggle/kaggle.X1.train.txt', header=None)
    Y = pd.read_csv('../data/kaggle/kaggle.Y.train.txt', header=None)

    smallestD = 0;
    smallestP = 0;
    smallestMSE = 10000000;

    d = 15
    print "Current Depth", d
    for p in range(1, 2):
        print "Current MinParent", p
        clf = tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=d, min_samples_split=2 ** p)
        scores = cross_validation.cross_val_score(clf, X, Y, cv=5)
        print scores
        temp = scores.mean()
        if smallestMSE > temp:
            print "P|D|Temp - ", p, d, temp
            smallestP = p
            smallestD = d
            smallestMSE = temp

    print "P|D|Temp Final ** - ", smallestP, smallestD, smallestMSE


def both():
    X = pd.read_csv('../data/kaggle/kaggle.X1.train.txt', header=None)
    Y = pd.read_csv('../data/kaggle/kaggle.Y.train.txt', header=None)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.25, random_state=42)

    est1 = RandomForestRegressor(n_estimators=2000, max_depth=3, min_samples_leaf=300)
    est = GradientBoostingRegressor(n_estimators=2000, max_depth=3, min_samples_leaf=300)

    est.fit(Xtr, Ytr)
    est1.fit(Xtr, Ytr)

    pred = est.predict(Xte)
    temp = mean_squared_error(Yte, pred)
    pred1 = est1.predict(Xte)
    temp1 = mean_squared_error(Yte, pred1)

    print temp
    print temp1


def boost():
    X = pd.read_csv('../data/kaggle/kaggle.X1.train.txt', header=None)
    Y = pd.read_csv('../data/kaggle/kaggle.Y.train.txt', header=None)
    Xtest = pd.read_csv('../data/kaggle/kaggle.X1.test.txt', header=None)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.25, random_state=42)

    est = GradientBoostingRegressor(n_estimators=2000, max_depth=3, min_samples_leaf=300)
    est = GradientBoostingRegressor(n_estimators=50, max_depth=3, min_samples_leaf=500, warm_start=True)
    est.fit(X, Y)

    pred = est.predict(Xtest)
    pd.Series(pred).to_csv('pyprediction.csv', header=['Prediction'], index=True, index_label='ID')


def boost1():
    X = pd.read_csv('../data/kaggle/kaggle.X1.train.txt', header=None)
    Y = pd.read_csv('../data/kaggle/kaggle.Y.train.txt', header=None)
    Xtest = pd.read_csv('../data/kaggle/kaggle.X1.test.txt', header=None)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.25, random_state=42)

    for i in range(30, 70, 2):
        est = GradientBoostingRegressor(n_estimators=i, max_depth=3, min_samples_leaf=500, warm_start=True)
        est.fit(Xtr, Ytr)
        Yhat = est.predict(Xte)
        temp1 = mean_squared_error(Yte, Yhat)
        print "The Result: ", i, "-", temp1

        # pd.Series(pred).to_csv('pyprediction.csv', header=['Prediction'], index=True, index_label='ID')


def boost2():
    minimum_mse = 1000000000
    min_depth = 0
    X = pd.read_csv('../data/kaggle/kaggle.X1.train.txt', header=None)
    Y = pd.read_csv('../data/kaggle/kaggle.Y.train.txt', header=None)
    Xtest = pd.read_csv('../data/kaggle/kaggle.X1.test.txt', header=None)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.25, random_state=42)

    for estimators in range(700, 2000, 100):
        print "For estimators: ", estimators
        for i in range(6, 8):
            print "For max_depth: ", i
            est = GradientBoostingRegressor(n_estimators=estimators, max_depth=i, min_samples_leaf=500, warm_start=True)
            est.fit(Xtr, Ytr)
            Yhat = est.predict(Xte)
            current_mse = mean_squared_error(Yte, Yhat)
            print "For MaxDepth:", i, ", MSE:", current_mse
            if minimum_mse > current_mse:
                minimum_mse = current_mse
                min_depth = i
                min_estimator = estimators
                est2 = GradientBoostingRegressor(n_estimators=min_estimator, max_depth=min_depth, min_samples_leaf=500,
                                                 warm_start=True, verbose=True)
                est2.fit(X, Y)

    print "** minimum_mse: ", minimum_mse
    print "** min_depth: ", min_depth
    print "** min_estimator: ", min_estimator

    pred = est2.predict(Xtest)
    s = pd.Series(pred)
    s.index = s.index + 1
    s.to_csv('pyprediction.csv', header=['Prediction'], index=True, index_label='ID')


def boost3():
    X = pd.read_csv('../data/kaggle/kaggle.X1.train.txt', header=None)
    Y = pd.read_csv('../data/kaggle/kaggle.Y.train.txt', header=None)
    Xtest = pd.read_csv('../data/kaggle/kaggle.X1.test.txt', header=None)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.25, random_state=42)

    est = GradientBoostingRegressor(n_estimators=2500, max_depth=8, min_samples_leaf=500, warm_start=True)
    est.fit(Xtr, Ytr)
    Yhat = est.predict(Xte)

    current_mse = math.sqrt(mean_squared_error(Yte, Yhat))
    print "-=- MSE:", current_mse

    pred = est.predict(Xtest)
    s = pd.Series(pred)
    s.index += 1
    s.to_csv('grad-boost-2000-depth-8-prediction.csv', header=['Prediction'], index=True, index_label='ID')

# main()
# cv()
# single()
# boost1()
warnings.simplefilter("ignore")
# boost2()
boost3()

'''
900 learner -
Your submission scored 0.59752, which is not an improvement of your best score. Keep trying!
'''

'''
usr/local/bin/python /Users/varadmeru/uci-related/uci-courses/CS273a-Introduction-to-Machine-Learning/HWs/HW4-complete/SKLearnTrees/SKLearnTree.py
Current Depth 8
Current MinParent 6
[ 0.37672004  0.36609664  0.38814851  0.39401899  0.39561548]
P|D|Temp -  6 8 0.384119931844
Current MinParent 7
[ 0.38212788  0.37147058  0.3880288   0.39820027  0.39555748]
Current MinParent 8
[ 0.38072636  0.37624433  0.38594976  0.40338973  0.39912784]
Current MinParent 9
[ 0.38931777  0.37871792  0.38864318  0.40514278  0.39938464]
Current MinParent 10
[ 0.3860388   0.3733171   0.38650367  0.39607105  0.39008914]
Current MinParent 11
[ 0.36907893  0.36014552  0.36775718  0.38340257  0.38204262]
P|D|Temp -  11 8 0.372485363426
Current Depth 9
Current MinParent 6
[ 0.37445927  0.36081415  0.37335572  0.38898359  0.38941843]
Current MinParent 7
[ 0.38107186  0.36906809  0.37678031  0.39567872  0.38972023]
Current MinParent 8
[ 0.38372145  0.37864335  0.38178046  0.40185151  0.39718832]
Current MinParent 9
[ 0.38967745  0.38216381  0.38932146  0.40772159  0.40024211]
Current MinParent 10
[ 0.38547754  0.3763311   0.38887408  0.39852398  0.39097456]
Current MinParent 11
[ 0.36946489  0.36194918  0.36801996  0.38383503  0.38207836]
P|D|Temp -  6 10 0.36439424583
Current Depth 10
Current MinParent 6
[ 0.3639835   0.3448174   0.35914451  0.37397014  0.38005568]
Current MinParent 7
[ 0.37619389  0.35521676  0.36401703  0.38429615  0.38608848]
Current MinParent 8
[ 0.38132966  0.3715173   0.38005127  0.39523204  0.39861038]
Current MinParent 9
[ 0.39123725  0.38231715  0.38927318  0.40647391  0.40199162]
Current MinParent 10
[ 0.3864174   0.37785087  0.38954152  0.39937338  0.39359568]
Current MinParent 11
[ 0.3697665   0.36268285  0.36801976  0.38383362  0.38208706]
Current Depth 11
Current MinParent 6
[ 0.34653549  0.32387332  0.34596657  0.35207932  0.36266599]
P|D|Temp -  6 11 0.346224139614
Current MinParent 7
[ 0.36465857  0.34284938  0.35579979  0.36896482  0.38076456]
Current MinParent 8
[ 0.37679314  0.36491099  0.37804926  0.39085827  0.39758235]
Current MinParent 9
[ 0.39124192  0.38245318  0.39029828  0.40429468  0.40250769]
Current MinParent 10
[ 0.3871654   0.37820509  0.39006171  0.39995458  0.39361263]
Current MinParent 11
[ 0.36976446  0.3630553   0.36803066  0.3838313   0.3820973 ]
P|D|Temp -  6 11 0.346224139614

Process finished with exit code 0
'''

'''
/usr/local/bin/python /Users/varadmeru/uci-related/uci-courses/CS273a-Introduction-to-Machine-Learning/HWs/HW4-complete/SKLearnTrees/SKLearnTree.py
Current Depth 11
Current MinParent 1
[ 0.29934024  0.27167141  0.31626521  0.30024957  0.32712682]
P|D|Temp -  1 11 0.302930650056
Current MinParent 2
[ 0.30150262  0.27173987  0.31616145  0.29854738  0.32887782]
Current MinParent 3
[ 0.3103397   0.27795382  0.31832911  0.30768302  0.33123944]
Current MinParent 4
[ 0.32235825  0.28821304  0.32662863  0.31930868  0.34311436]
Current MinParent 5
[ 0.33293032  0.30668519  0.33108364  0.34323451  0.35194748]
Current MinParent 6
[ 0.34680808  0.32361811  0.34505702  0.35302631  0.36203478]
Current MinParent 7
[ 0.36443843  0.34233828  0.35592954  0.36896482  0.38121036]
Current MinParent 8
[ 0.37800339  0.36335132  0.37778662  0.39085827  0.39778936]
Current MinParent 9
[ 0.39124192  0.38245318  0.39003564  0.40429468  0.40250769]
P|D|Temp Final ** -  1 11 0.302930650056

Process finished with exit code 0
'''