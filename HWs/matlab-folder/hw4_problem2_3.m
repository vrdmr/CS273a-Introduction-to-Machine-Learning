%% Homework 4 - Problem 1 
% Author: Varad Meru (vmeru@uci.edu)
% Course: CS 273, Machine Learning (http://sli.ics.uci.edu/Classes/2015W-273a)
% Homework Description - http://sli.ics.uci.edu/Classes/2015W-273a?action=download&upname=HW3.pdf
% Started on 26th Jab, 15.

email_data = [0 0 1 1 0 -1;
     1 1 0 1 0 -1;
     0 1 1 1 1 -1;
     1 1 1 1 0 -1;
     0 1 0 0 0 -1;
     1 0 1 1 1 1;
     0 0 1 0 0 1;
     1 0 0 0 0 1;
     1 0 1 1 0 1;
     1 1 1 1 1 -1];
%{
learner = treeClassify();
learner = setClasses(learner, uniqClasses);
train(learner,X,y);
 %}
y = email_data(:,end);
X = email_data(:,1:end-1);
classes = unique(y);
[uniqClasses, numUniqClasses] = count_unique(y);

T=classregtree(X,y,'splitmin',1, 'names',{'X1' 'X2' 'X3' 'X4' 'X5'});
T,
h=figure;
view(T);