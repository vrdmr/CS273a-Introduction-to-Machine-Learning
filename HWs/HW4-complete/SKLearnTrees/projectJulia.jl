Pkg.add("GLM")

Xtr = readtable("/Users/varadmeru/uci-related/uci-courses/CS273a-Introduction-to-Machine-Learning/HWs/HW4-complete/SKLearnTrees/kaggle.X1.train.txt")
Ytr = readtable("/Users/varadmeru/uci-related/uci-courses/CS273a-Introduction-to-Machine-Learning/HWs/HW4-complete/SKLearnTrees/kaggle.Y.train.txt")
Xte = readtable("/Users/varadmeru/uci-related/uci-courses/CS273a-Introduction-to-Machine-Learning/HWs/HW4-complete/SKLearnTrees/kaggle.X1.test.txt")
coefs = linreg(Xtr, Ytr)
