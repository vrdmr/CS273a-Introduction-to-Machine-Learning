%% Homework 5 - Problem 3: Eigen Faces
clc;close all;clear all;
rand('seed',0);
X = load('data/faces.txt'); % load face dataset


%%
for i = 1;
    img = reshape(X(i,:),[24 24]); % convert vectori
end;
imagesc(img);
axis square; 
colormap gray;

%% Faces A
mu = mean(X);
X0 = bsxfun(@minus,X,mu);
size(mu)

%%
n = size(X0,1);
[U,S,V] = svd(X0);
W = U * S;
size(W)