%% Homework 4 - Problem 3 
% Author: Varad Meru (vmeru@uci.edu)
% Course: CS 273, Machine Learning (http://sli.ics.uci.edu/Classes/2015W-273a)
% Homework Description - http://sli.ics.uci.edu/Classes/2015W-273a?action=download&upname=HW3.pdf
% Started on 26th Jab, 15.

% Problem a - Decision trees on Kaggle

weatherX=load('data/kaggle/kaggle.X1.train.txt');     % load the text file
weatherY=load('data/kaggle/kaggle.Y.train.txt');     % load the text file
Xeval = load('data/kaggle/kaggle.X1.test.txt');
% feature_names=load('data/kaggle/kaggle.X1.names.txt');
X = weatherX;
Y = weatherY;

[X, Y] = shuffleData(X,Y);
[Xtr, Xte, Ytr, Yte] = splitData(X,Y, .75); % split data into 75/25 train/test
%%
dt = treeRegress(Xtr,Ytr, 'maxDepth',20);
mse(dt,Xte,Yte) 
% ans = 0.7367

dt = treeRegress(Xtr,Ytr, 'maxDepth',15);
mse(dt,Xte,Yte)
% ans = 0.6384

%%
% Dataset X,Y
mu = mean(Y); % Often start with constant ?mean? predictor 
dY = Y - mu; % subtract this prediction away
Nboost = 20;
for k=1:Nboost,
  Learner{k} = treeRegress(X,dY, 'maxDepth',15);
  alpha(k) = 1;  % alpha: a ?learning rate? or ?step size?
  % smaller alphas need to use more classifiers, but tend to
  %   predict better given enough of them
  % compute the residual given our new prediction
  dY = dY - alpha(k) * predict(Learner{k}, X);
end;

% Test data Xtest
[Ntest,D] = size(Xeval);
predict = zeros(Ntest,1);      % Allocate space
for k=1:Nboost,                % Predict with each learner
  predict = predict + alpha(k)*predict(Learner{k}, Xeval);
end;

X * Y
X .* Y
