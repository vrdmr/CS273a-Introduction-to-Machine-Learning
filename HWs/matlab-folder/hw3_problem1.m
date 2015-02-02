%% Problem 1
% Author: Varad Meru (vmeru@uci.edu)
% Course: CS 273, Machine Learning (http://sli.ics.uci.edu/Classes/2015W-273a)
% Homework Description - http://sli.ics.uci.edu/Classes/2015W-273a?action=download&upname=HW3.pdf
% Started on 26th Jab, 15.

iris=load('data/iris.txt');     % load the text file
Y = iris(:,end);           % target value is last column
X = iris(:,1:end-1);       % features are other columns
features = char('Sepal length','Sepal width','Petal length','Petal width','Species');
features_short = char('SL','SW','PL','PW','SP');

% Vertically Splitting the dataset. Keeping the first 2 columns for the
% perceptron.
Xs = X(:,1:2);

% Shuffling and Rescaling X
[Xs Y] = shuffleData(Xs,Y);
Xs  = rescale(Xs);

% get class 0 vs 1
XA = Xs(Y<2,:); 
YA=Y(Y<2);
% get class 1 vs 2
XB = Xs(Y>0,:); 
YB=Y(Y>0);

%% Problem a
h=figure;
scatter(XA(:,1), XA(:,2), 50, YA,'filled')
saveas(h,'scatter-classes1.jpg','jpg');
h=figure;
scatter(XB(:,1), XB(:,2), 50, YB,'filled')
saveas(h,'scatter-classes2.jpg','jpg');

%% Problem b
learner = logisticClassify2();
learner=setClasses(learner, unique(YA));
wts = [.5 1 -.25];
learner=setWeights(learner, wts);

h=figure;
plot2DLinear(learner,XA,YA);
saveas(h,'log1.jpg','jpg');

h=figure;
plot2DLinear(learner,XB,YB);
saveas(h,'log2.jpg','jpg');

%% Problem c
yte = predict(learner,XA);
error = errorTrain(YA,yte);
% error = 0.0505

yte = predict(learner,XB);
error = errorTrain(YB,yte);
% error = 0.5455

%% Problem e-1
h=figure;
train(learner,XA, YA);
saveas(h,'logistic1.jpg','jpg');
%% Problem e-2
h=figure;
train(learner,XA, YA);
saveas(h,'logistic2.jpg','jpg');

plotClassify2D(learner, XA, YA);

%% Problem b
learner = logisticClassify2();
learner=setClasses(learner, unique(YB));
wts = [.5 1 -.25];
learner=setWeights(learner, wts);
train(learner,XB, YB);

h=figure;
plotClassify2D(learner, XB, YB);
saveas(h,'logisticclassifyxb.jpg','jpg');
%plotClassify2D(learner, XA, YA);
%saveas(h,'logisticclassify.jpg','jpg');