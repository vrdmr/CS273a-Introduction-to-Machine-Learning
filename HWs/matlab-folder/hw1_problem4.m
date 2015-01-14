%% Problem 4
% Author: Varad Meru (vmeru@uci.edu)
% Course: CS 273, Machine Learning (http://sli.ics.uci.edu/Classes/2015W-273a)
% Homework Description - http://sli.ics.uci.edu/Classes/2015W-273a?action=download&upname=HW1.pdf
% Started on 08th Jab, 15.
iris=load('data/iris.txt');     % load the text file
y = iris(:,end);           % target value is last column
X = iris(:,1:end-1);       % features are other columns
features = char('Sepal length','Sepal width','Petal length','Petal width','Species');
features_short = char('SL','SW','PL','PW','SP');
s = RandStream('mt19937ar','Seed',1);
% Changed the shuffleData function to take a seed from the user. If not
% provided, then it generates it randomly.
[X, y] = shuffleData(X,y,s);

X = X(:,1:2);
[Xtr,Xte,Ytr,Yte] = splitData(X,y, .75);

%% Part(a)
% Splitting by class

[tempUniq, numTemp] = count_unique(Ytr);
mean_matrix = zeros(size(tempUniq,1), size(Xtr,2));
for i = 1:size(tempUniq,1);
    temp = Xtr(find(Ytr==tempUniq(i,:)),:); % All the indexes of 
    disp(strcat('Class:',num2str(i)))
    disp('--------------------------------')
    mean_matrix (i,:) = mean(temp); % Mean Stored for future use
    disp('Mean of the class')
    disp(mean_matrix (i,:));
    disp('Covariance matrix')
    disp(cov(temp)); % Creating and displaying the Covariance matrix of the class data.
end;
%% Part (b)
% 
h = figure;
scatter(Xtr(:,1), Xtr(:,2), 50, Ytr, 'filled');
saveas(h,'scatter-bayes.jpg','jpg');

%{
[tempUniq, numTemp] = count_unique(Ytr);
mean_matrix = zeros(size(tempUniq,1), size(Xtr,2));
for i = 1:size(tempUniq,1);
    temp = Xtr(find(Ytr==tempUniq(i,:)),:); % All the indexes of 
    mean_matrix (i,:) = mean(temp); % Mean Stored for future use
    cov(temp); % Creating and displaying the covarianve matrix of the class data.
end;
%}

% gauss = gaussBayesClassify(Xtr, Ytr);

%% Part(c) - stable
% Evaluate each point of feature space and predict the class
[tempUniq, numTemp] = count_unique(Ytr);
mean_matrix = zeros(size(tempUniq,1), size(Xtr,2));
color = {'blue','green','yellow'};
h = figure;
scatter(Xtr(:,1),Xtr(:,2), 50, Ytr, 'filled');
hold on;
for i = 1:size(tempUniq,1);
    temp = Xtr(find(Ytr==tempUniq(i,:)),:); % All the indexes of 
    mean_matrix (i,:) = mean(temp); % Mean Stored for future use
    plotGauss2D(mean(temp),cov(temp), color{i});
    hold on;
end;
hold off;
saveas(h,'scatter-bayes-2.jpg','jpg');

%% Part(c) - extended
% Evaluate each point of feature space and predict the class
[tempUniq, numTemp] = count_unique(Ytr);
cmap=jet(256); 
clim=unique(tempUniq)';
mean_matrix = zeros(size(tempUniq,1), size(Xtr,2));
color = {'blue','green','yellow'};
figure;
% scatter(Xtr(:,1),Xtr(:,2), 50, Ytr, 'filled');
hold on;
for i =1:size(tempUniq);
    col= fix(floor((i-min(clim))/(max(clim)-min(clim)))*255+1);
    
    temp = Xtr(find(Ytr==tempUniq(i,:)),:); % All the indexes of 
    mean_matrix (i,:) = mean(temp); % Mean Stored for future use
    
    plot(Xtr(find(Ytr==tempUniq(i,:)),1),Xtr(find(Ytr==tempUniq(i,:)),2),'o','markersize',7,'color',cmap(col,:),'markerfacecolor',cmap(col,:)); 
    plotGauss2D(mean(temp),cov(temp), 'k','linewidth',3,'Color',cmap(col,:));
    hold on;
end;
hold off;
%% Part(d)
h = figure;
bc = gaussBayesClassify( Xtr, Ytr );
plotClassify2D(bc, Xtr, Ytr);
saveas(h,'bayes-2.jpg','jpg');

%% Part(e)
bc = gaussBayesClassify( Xtr, Ytr );
YtrHat = predict( bc, Xtr );
[errorsCount, errorRate] = errorTrainBayes(Ytr, YtrHat);

bcTe = gaussBayesClassify( Xte, Yte );
YteHat = predict( bcTe, Xte );
[errorsCount, errorRate] = errorTrainBayes(Yte, YteHat);

%% Part(f)
bc = gaussBayesClassify( Xtr, Ytr );
YtrHat = predict( bc, Xtr );
[errorsCountTr, errorRateTr] = errorTrainBayes(Ytr, YtrHat);
[errorsCountTr, errorRateTr]

bcTe = gaussBayesClassify( Xte, Yte );
YteHat = predict( bcTe, Xte );
[errorsCountTe, errorRateTe] = errorTrainBayes(Yte, YteHat);
[errorsCountTe, errorRateTe]

%% TEMP Code

%{
N=256;    % density of evaluation
ax=axis;
X1 = linspace(ax(1),ax(2),N); X1sp=X1'*ones(1,N);
X2 = linspace(ax(3),ax(4),N); X2sp=ones(N,1)*X2;
Xfeat = [X1sp(:),X2sp(:)];


cmap=jet(256);
clim=unique(Ytr)';
cmapshade = cmap*.4+.6;
colormap(cmapshade);
% plot decision values for the space in "faded" color
imagesc(X1,X2,reshape(pred,N,N)',[clim(1) clim(end)]); axis xy; hold on; colormap(cmapshade);
theta = [0:.01:2*pi]';
circle = [sin(theta), cos(theta)];
ell = circle * sqrtm(gCov);
ell = ell + ones(size(ell,1),1)*gMean;

[tempUniq, numTemp] = count_unique(Ytr);
mean_matrix = zeros(size(tempUniq,1), size(Xtr,2));
figure;
scatter(Xtr(:,1),Xtr(:,2), 50, Ytr, 'filled');
hold on;
for i = 1:size(tempUniq,1);
    col= fix((c-min(clim))/(max(clim)-min(clim))*255+1);
    temp = Xtr(find(Ytr==tempUniq(i,:)),:); % All the indexes of 
    mean_matrix (i,:) = mean(temp); % Mean Stored for future use
    plotGauss2D(mean(temp),cov(temp),['k' 'x'], ell(:,1), ell(:,2), 'k','linewidth',3,'Color',cmap(col,:) , varargin{:})
    hold on;
end;
hold off;

%}