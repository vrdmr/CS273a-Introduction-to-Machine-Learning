%% Problem 3
% Author: Varad Meru (vmeru@uci.edu)
% Course: CS 273, Machine Learning (http://sli.ics.uci.edu/Classes/2015W-273a)
% Homework Description - http://sli.ics.uci.edu/Classes/2015W-273a?action=download&upname=HW1.pdf
% Started on 08th Jab, 15.

%{
In Order to reduce my email load, I decide to implement a machine learning
algorithm to decide whether or not I should read my email, or simply file
it away instead. TO train my model, I obtain the following data set of
binary-valued features about each email, including: 
author: known?
email: long or short?
keywords - research/grade/lottery?

Last column is the result: y = +1 for "read", y = -1 for "discard"
%}

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
 
y = email_data(:,end);
X = email_data(:,1:end-1);
classes = unique(y);

[uniqClasses, numUniqClasses] = count_unique(y);
% class_prob = zeros(size(uniqClasses)); % We'll calculate it later.
class_prob = numUniqClasses ./ sum(numUniqClasses);

class_count = zeros(size(uniqClasses,1), size(X,2));

% Counting the occurances of Xi for different values of the classes
% For each feature - 
for xi = 1:size(X,2);
    [uniqX, numUniqX] = count_unique(X(:,xi));
    for xn = 1:size(X,1);
        if X(xn,xi) == 1;
            class_count(find(uniqClasses == y(xn)) , xi) = class_count(find(uniqClasses == y(xn)) , xi) + 1;
        end;
    end;
end;

% Gets the probabilities of the ones cases, where the data is 1.
probabilities_of_one_cases = zeros(size(class_count));
for col = 1:size(probabilities_of_one_cases, 2);
    for row = 1:size(probabilities_of_one_cases, 1);
        probabilities_of_one_cases(row,:) = class_count(row,:) ./ numUniqClasses (row);
    end;
end;

% Gets the probabilities of the zero cases, where the data is 0.
probabilities_of_zero_cases = 1 - probabilities_of_one_cases;

%disp('probabilities_of_one_classes');
[probabilities_of_one_cases, uniqClasses];
%disp('probabilities_of_zero_classes');
[probabilities_of_zero_cases, uniqClasses];

%% Part (b AND c)

% 
% P(y|X) = p(X|y) p(y) / P(X)
% P(y = -1 |X) = p(X|y = -1) p(y = -1) / P(X)

% P(y = -1 |X) = 
% p(X|y = -1) p(y = -1)
% ------------------------------------
%(p(X|y=-1) p(y=-1) + p(X|y=1) p(y=1))

% P(y = 1 |X) = 
% p(X|y = 1) p(y = 1)
% ------------------------------------
%(p(X|y=-1) p(y=-1) + p(X|y=1) p(y=1))

% input = [0, 0, 0, 0, 0];
input = [1, 1, 0, 1, 0];

% finding the classes for -1
productClass1 = 1;
for i = 1:size(input,2); 
    if input(:,i) == 1;
       productClass1 = productClass1 *  probabilities_of_one_cases(1,i);
    else
       productClass1 = productClass1 *  probabilities_of_zero_cases(1,i);
    end;
end;

% finding the classes for 1
productClass2 = 1;
for i = 1:size(input,2);
    if input(:,i) == 1;
       productClass2 = productClass2 *  probabilities_of_one_cases(2,i);
    else
       productClass2 = productClass2 *  probabilities_of_zero_cases(2,i);
    end;
end;

probClass1 = productClass1 * class_prob(1,:) / (productClass1 * class_prob(1,:) + productClass2 * class_prob(2,:));
probClass2 = productClass2 * class_prob(2,:) / (productClass1 * class_prob(1,:) + productClass2 * class_prob(2,:));

%{
probClass1 =
     1
probClass2 =
     0
%}
