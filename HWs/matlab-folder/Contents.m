% Matlab-based machine learning tools for teaching

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simple data generation functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dataGauss(N0,N1,mu0,mu1,Sig0,Sig1) : generate two-class Gaussian data 
% dataMouse([symbolSize])            : generate two-class data from mouse clicks


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basic data manipulation functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% shuffleData(X [,Y])            : randomly permute (reorder) a data set
% splitData(X, Y, trainFraction) : split data into training/validation sets
% crossValidate(X,Y,nFolds,iFold): split data for n-fold cross validation
% bootstrapData(X,Y,nBoot)       : resample (bootstrap) data set
% imputeMissing(X,method,R)      : "fill in" missing (nan) values in data
%
% toIndex(Y,values)              : convert discrete class Y into index (1..K) representation
% fromIndex(Y,values)            : convert class 1..K into discrete class values for Y 
% to1ofK(Y,values)               : convert discrete class Y into 1-of-K representation
% from1ofK(Y,values)             : convert 1-of-K Y into discrete representation


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Features and other data transformations
% All functions are F(required [, optional]); basic usage generates the transform with the first
%   call, then re-applies it in subsequent calls:  [X1new, Params] = F(X1);  [X2new]=F(X2,Params); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rescale(X [, scale])           : rescale data to unit variance
% whiten(X [, mu,sig])           : whitens X to be zero mean, uncorrelated and unit variance
%
% fsubset(X, K [, feat])         : select subset of features 
% fproject(X, K [, proj])        : random projection of features 
% fsvd(X, K [, T])               : project on top-K principal components (computed w/ SVD)
% fhash(X, K [, hash])           : random hash of features from data
% flda(X, Y, K, T)               : reduce # features to K using (multiclass) linear discrim analy
% fpoly(X, degree, const)        : create extended polynomial features 
% fkitchensink(X, K, type [,W])  : random kitchen sink features from data


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plotRegress1D((learner,X,Y [,pre,Color]) : plot 1D feature and 1D target with regression line
% plotClassify2D((learner,X,Y [,pre])      : plot 2D data and classification boundaries
% plotGauss2D( gMean, gCov, coloris, ...)  : plot Gaussian ellipses on 2D data
%
% plotPairs(X [,Y])                        : standard "all feature pairs" scatterplot 
% histy(X,Y,options)                       : multi-class overlapping histograms


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Unsupervised learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% kmeans(X,K,init)   : cluster data into K clusters via k-means, with initial cluster centers "init"
% emClust(X,K,...)   : Gaussian mixture Expectation-Maximization clustering
% agglomClust(X,...) : Agglomerative clustering 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Supervised learning methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% knnClassify         : k-nearest neighbor & weighted local classification
% knnRegress          : k-nearest neighbor & weighted local regression

% gaussBayesClassify  : Bayes classifier with Gaussian class-conditional distributions

% treeClassify        : decision tree classifiers (w/ random forest support)
% treeRegress         : regression trees (w/ random forest support)

% linearRegress       : linear regression

% Linear classifiers:
%   Perceptron training
%   Logistic MSE training
%   Logistic Neg Log-Likelihood training
%   Linear SVM / Hinge-loss training

% nnetClassify        : basic MLP / neural network for classification
% nnetRegress         : basic MLP / neural network for regression


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Meta-learning functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% baggedClassify      : bagged ensemble of classifiers
% baggedRegress       :  "" "" of regressors

% gradboost           : gradient boosting training for regression

% adaboost            : adaboost training for classification


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Miscellaneous helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fig(n)              : open / raise figure n without switching focus



