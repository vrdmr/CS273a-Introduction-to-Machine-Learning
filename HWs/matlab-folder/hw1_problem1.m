%% Problem 1
% Author: Varad Meru (vmeru@uci.edu)
% Course: CS 273, Machine Learning (http://sli.ics.uci.edu/Classes/2015W-273a)
% Homework Description - http://sli.ics.uci.edu/Classes/2015W-273a?action=download&upname=HW1.pdf
% Started on 08th Jab, 15.

% Fetching the dataset and separating it into X and Y.
iris=load('data/iris.txt');     % load the text file
y = iris(:,end);           % target value is last column
X = iris(:,1:end-1);       % features are other columns
features = char('Sepal length','Sepal width','Petal length','Petal width','Species');
features_short = char('SL','SW','PL','PW','SP');
whos

%% Part (a)
% In this problem, we will explore some basic statistics and visualizations of an example data set.

% Part (a) - Use size(X,2) to get the number of features, and size(X,1) to get the number of data points.
disp(size(X, 1));
disp(size(X, 2));

%% Part (b)
% For each feature, plot a histogram ("hist") of the data values

for f=1:size(X, 2);
    h=figure;
    %subplot(1,4,f)
    h1=histogram(X(:,f)); % histogram is preferred over hist - matlab documentation
    title(strcat('Feature:- ',features(f,:)))
    %hold on;
    saveas(h,strcat('histogram',num2str(f)),'jpg');
end;
%hold off;


% All in one
h=figure;
for f=1:size(X, 2);
    histogram(X(:,f));
    hold on;
end;
hold off;
saveas(h,'histogram5.jpg','jpg');

%% Part (c)
% Compute the mean of the data points for each feature (mean)
disp('Mean');
meanX = mean(X);
for f=1:size(X, 2);
    disp(strcat({'Feature:- '}, features(f,:),{': '} , num2str(meanX(:,f))));
end;

%% Part (d)
% Compute the variance and standard deviation of the data points for each feature
disp('Standard Deviations')
stdX = std(X);
for f=1:size(X, 2);
    disp(strcat({'Feature:- '}, features(f,:),{': '} , num2str(stdX(:,f))));
end;

disp('Variances')
varX = var(X);
for f=1:size(X, 2);
    disp(strcat({'Feature:- '}, features(f,:),{': '} , num2str(varX(:,f))));
end;

%% Part (e)
% "Normalize" the data by subtracting the mean value from each feature, 
% and dividing by its standard deviation. (This will make the data zero-mean and 
% unit variance.) Show your code. Note: you can do this with a for-loop (easy, but 
% slow in Matlab), or in a "vectorized" form using repmat or bsxfun (faster, but 
% harder to read).
normX = bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), std(X));

%{
One More way to do this (more understandable way) -

meanXMat = repmat(meanX, size(X,1),1);
stdXMat = repmat(stdX, size(X,1),1);
norX1 = X - meanXMat;
normX = norX1 ./ stdXMat;
%}

%% Part (f)
% For each pair of features (1,2), (1,3), and (1,4), plot a 
% scatterplot (see "plot" or "scatter") of the feature values, colored according 
% to their target value (class). (For example, plot all data points with y = 0 
% as blue, y = 1 as green, etc.) Note: if you wish to overlay several plot 
% commands, use "hold on" before subsequent plots; to stop this behavior, 
% use "hold off".
h=figure;
i = 1;
for f=2:size(X, 2);
    subplot(1,3,i);
    h1 = scatter(X(:,1), X(:,f), 50, y, 'filled');
    i = i+1;
    title(strcat({'X: '},features_short(1,:), {',Y:'}, features_short(f,:)))
    
    %{
    subplot(2,3,i);
    h2 = scatter(X(:,f), X(:,1), 50, y, 'filled');
    i = i+1;
    title(strcat({'X: '},features(f,:), {',Y:'}, features(1,:)))
    %}
    
    hold on;
    
end;
hold off;
saveas(h,'plots.jpg','jpg');

% You'll need 2-for loops for a complete scatter plot. Work on that later.

%% TEMP CODE
%{
for f=1:size(X, 2);
h=hist(X(:,f))
saveas(h,strcat('plots',f))
end;

for f=1:size(X, 2);
histogram(X(:,f)); 
hold on;
end;

for f=1:size(X, 2);
subplot(2,2,f)
h=histogram(X(:,f));
title(strcat('Feature: ', num2str(f)))
hold on;
end;
%}
