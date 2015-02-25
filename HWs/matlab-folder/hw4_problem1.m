%% Homework 4 - Problem 1 
% Author: Varad Meru (vmeru@uci.edu)
% Course: CS 273, Machine Learning (http://sli.ics.uci.edu/Classes/2015W-273a)
% Homework Description - http://sli.ics.uci.edu/Classes/2015W-273a?action=download&upname=HW3.pdf
% Started on 26th Jab, 15.

% Problem 1 part b -  Primal form

iris=load('data/iris.txt');     % load the text file
X = iris(:,1:2);       % features are other columns
Y = iris(:,end);           % target value is last column
XA = X(Y<2,:);
YA=Y(Y<2);       % get class 0 vs 1
YB(YA == 0) = -1;
YB(YA == 1) = 1;

XA_CLASS_0 = XA(find(YB==-1),:);
XA_CLASS_1 = XA(find(YB==1),:);

h = figure;
axis([4,7,1.5,5])
hold on;
plot(XA_CLASS_0(:,1),XA_CLASS_0(:,2),'or');
plot(XA_CLASS_1(:,1),XA_CLASS_1(:,2),'+b');
hold off;
saveas(h,'primal1.jpg','jpg');
numOfExamples = size(XA,1);
numOfAttributes = size(XA,2);
H = eye( numOfAttributes + 1);
H(numOfAttributes+1,numOfAttributes+1)=0;
f=zeros(numOfAttributes+1,1);

Z = [XA ones(numOfExamples,1)];
A=-diag(YB)*Z;
c=-1*ones(numOfExamples,1);
w=quadprog(H,f,A,c);

w1=w(1,1);
w2=w(2,1);
b=w(3,1);

Y1=-(w1*XA+b)/w2;  %Seperating hyperplane

h = figure;
axis([4,7,1.5,5])
hold on;
plot(XA_CLASS_0(:,1),XA_CLASS_0(:,2),'or');
plot(XA_CLASS_1(:,1),XA_CLASS_1(:,2),'+b');
%%% Code to plot the SVM margins goes here! %%%
plot(XA,Y1,'k-');

YUP=(1-w1*XA-b)/w2; %Margin
plot(XA,YUP,'m:');
YLOW=(-1-w1*XA-b)/w2; %Margin
plot(XA,YLOW,'m:');

title('Iris Dataset');
xlabel('Sepal length');
ylabel('Sepal width');
legend('Class 0','Class 1','SVM Hyperplane','Upper Margin','Lower Margin');

hold off;
saveas(h,'primal2.jpg','jpg');

%{
    wts =
  -16.7391    6.1716   -5.2282
%}

%% Problem 1 - Using Perceptron
% Author: Varad Meru (vmeru@uci.edu)
% Course: CS 273, Machine Learning (http://sli.ics.uci.edu/Classes/2015W-273a)
% Homework Description - http://sli.ics.uci.edu/Classes/2015W-273a?action=download&upname=HW3.pdf
% Started on 26th Jab, 15.

iris=load('data/iris.txt');     % load the text file
X = iris(:,1:2);       % features are other columns
Y = iris(:,end);           % target value is last column
XA = X(Y<2,:);
YA=Y(Y<2);       % get class 0 vs 1
YB(YA == 0) = -1;
YB(YA == 1) = 1;

XA_CLASS_0 = XA(find(YB==-1),:);
XA_CLASS_1 = XA(find(YB==1),:);

features = char('Sepal length','Sepal width','Petal length','Petal width','Species');
features_short = char('SL','SW','PL','PW','SP');

numOfExamples = size(XA,1);
numOfAttributes = size(XA,2);
H = eye( numOfAttributes + 1);
H(numOfAttributes+1,numOfAttributes+1)=0;
f=zeros(numOfAttributes+1,1);

Z = [XA ones(numOfExamples,1)];
A=-diag(YB)*Z;
c=-1*ones(numOfExamples,1);
w=quadprog(H,f,A,c);
% SVM part done.

learner = logisticClassify2();
learner=setClasses(learner, unique(YA));
wts =[w(3,1) w(1,1) w(2,1)];
learner=setWeights(learner, wts);

yte = predict(learner,XA);
error = errorTrain(YA,yte);

h=figure;
plot2DLinear(learner,XA,YA);
saveas(h,'primal4.jpg','jpg');
h=figure;
plotClassify2D(learner, XA, YA);
saveas(h,'primal3.jpg','jpg');

%% Problem 1 part b -  Dual form
clc;
close all;
clear all;
iris=load('data/iris.txt'); 

X = iris(:,1:2); Y=iris(:,end);
XA = X(Y<2,:); YA=Y(Y<2);
YA(YA==0) = -1;
XA_CLASS_0 = XA(find(YA==-1),:);
XA_CLASS_1 = XA(find(YA==1),:);

[numOfExamples, numOfAttributes] = size(XA);

K = (XA*XA');
H = K .* (YA*YA');
f = -ones(numOfExamples,1);

A = -eye(numOfExamples);
B = zeros(numOfExamples,1);

Aeq = [YA' ; zeros(numOfExamples-1,numOfExamples)];
beq = zeros(numOfExamples,1);
alpha = quadprog(H+eye(numOfExamples)*0.001,f,A,B,Aeq,beq);
% quadprog(H,f,A,B,Aeq,Beq,lb,ub,X0,options,varargin)

w = (alpha.*YA)'*XA;
XSuppVectors = XA(alpha>0.1,:);
YSuppVectors = YA(alpha>0.1);

b = (1/YSuppVectors(1)) - w*XSuppVectors(1,:)';
theta = [b,w];
learner=logisticClassify(); 
learner=setClasses(learner, unique(YA)); 
wts = [b,w];
learner=setWeights(learner, wts); 

h = figure;
plotClassify2D(learner, XA, YA);
saveas(h,'dual.jpg','jpg');

w1 = wts(:,2);
w2 = wts(:,3);
Y1=-(w1*XA+b)/w2;  %Seperating hyperplane

YUP=(1-w1*XA-b)/w2; %Margin
YLOW=(-1-w1*XA-b)/w2; %Margin

h = figure;
axis([4,7,1.5,5])
hold on;
plot(XA_CLASS_0(:,1),XA_CLASS_0(:,2),'or');
hold on;
plot(XA_CLASS_1(:,1),XA_CLASS_1(:,2),'+b');
hold on;
plot(XA,Y1,'k-');
hold on;
plot(XA,YUP,'m:');
hold on;
plot(XA,YLOW,'m:');
hold off;
title('Iris Dataset');
xlabel('Sepal length');
ylabel('Sepal width');
legend('Class 0','Class 1','SVM Hyperplane','and Decision Boundary','Upper Margin','Lower Margin');
saveas(h,'dual2.jpg','jpg');