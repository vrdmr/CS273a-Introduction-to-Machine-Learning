

clc;close all;clear all;
iris=load('data/iris.txt'); 
X = iris(:,1:2); Y=iris(:,end);
XA = X(Y<2,:); YA=Y(Y<2);
YA(YA==0) = -1;
[n,m] = size(XA);
%% Primal
H = eye(m+1);
H(3,3)=0;
f = zeros(m+1,1);
X1 = [XA ones(n,1)];
A = -diag(YA)*X1;
bienq = -1*ones(n,1);
w = quadprog(H,f,A,bienq);
w = w';
learner=logisticClassify(); 
learner=setClasses(learner, unique(YA)); 
wts = [w(3),w(1:2)];
learner=setWeights(learner, wts); 
figure,plotClassify2D(learner, XA, YA);

%% Dual
H = (XA*XA').*(YA*YA');
f = -ones(n,1);
A = -eye(n);
a = zeros(n,1);
B = [[YA'];[zeros(n-1,n)]];
b = zeros(n,1);
z = quadprog(H+eye(n)*0.001,f,A,a,B,b);
w = (z.*YA)'*XA;
XSV = XA(z>0.1,:);
YSV = YA(z>0.1);
% b = (1/YSV(1)) - w.*XSV(1,:);
b = (1/YSV(1)) - w*XSV(1,:)';
theta = [b,w];
learner=logisticClassify(); 
learner=setClasses(learner, unique(YA)); 
wts = [0.5 1 -0.25];
size(wts);
wts = [b,w];
learner=setWeights(learner, wts); 
figure,plotClassify2D(learner, XA, YA);