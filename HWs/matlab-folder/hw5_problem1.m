%% Homework 5 - Problem 1 Basics of Clustering
% Author: Varad Meru (vmeru@uci.edu)
% Course: CS 273, Machine Learning (http://sli.ics.uci.edu/Classes/2015W-273a)
% Homework Description - http://sli.ics.uci.edu/Classes/2015W-273a?action=download&upname=HW5.pdf
% Started on 05 Mar, 15.

% Problem a

iris=load('data/iris.txt');     % load the text file
X = iris(:,1:2);       % features are other columns
features = char('Sepal length','Sepal width','Petal length','Petal width','Species');
features_short = char('SL','SW','PL','PW','SP');
whos

scatter(X(:,1), X(:,2), 'filled');

%% Problem b

