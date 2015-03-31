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

f = figure;
scatter(X(:,1), X(:,2), 'filled');
saveas(f,'scatter.png','png');

%% Problem b
k = 5;
[z,c,sumd] = kmeans(X,k);
[z1,c1,sumd1] = kmeans(X,k,'k++');

f = figure;
plotClassify2D([],X,z);
saveas(f,'kmeans_k_5_simple.png', 'png');

f = figure;
plotClassify2D([],X,z1);
saveas(f,'kmeans_k_5_kpp.png', 'png');

k = 20;
[z,c,sumd] = kmeans(X,k);
[z1,c1,sumd1] = kmeans(X,k,'k++');

f = figure;
plotClassify2D([],X,z);
saveas(f,'kmeans_k_20_simple.png', 'png');

f = figure;
plotClassify2D([],X,z1);
saveas(f,'kmeans_k_20_kpp.png', 'png');

%% Problem c

k = 5;
Z = linkage(X,'single');
c = cluster(Z,'maxclust',k);
f = figure;
plotClassify2D([],X,c);
saveas(f,'linkage_single_5.png', 'png');
f = figure;
dendrogram(Z)
saveas(f,'dendogram_5.png', 'png');

Z = linkage(X,'complete');
c = cluster(Z,'maxclust',k);
f = figure;
plotClassify2D([],X,c);
saveas(f,'linkage_complete_5.png', 'png');

k = 20;
Z = linkage(X,'single');
c = cluster(Z,'maxclust',k);
f = figure;
plotClassify2D([],X,c);
saveas(f,'linkage_single_20.png', 'png');
f = figure;
dendrogram(Z)
saveas(f,'dendogram_20.png', 'png');

Z = linkage(X,'complete');
c = cluster(Z,'maxclust',k);
f = figure;
plotClassify2D([],X,c);
saveas(f,'linkage_complete_20.png', 'png');

%% Problem d
k = 5;
[zx,Tx,softx,llx] = emCluster(X,k);
f = figure;
plotClassify2D([],X,zx);
saveas(f,'emgm_5.png', 'png');

k = 20;
[zx,Tx,softx,llx] = emCluster(X,k);
f = figure;
plotClassify2D([],X,zx);
saveas(f,'emgm_20.png', 'png');

k = 5;
[zx,Tx,softx,llx] = emCluster(X,k,'k++');
f = figure;
plotClassify2D([],X,zx);
saveas(f,'emgm_5_kpp.png', 'png');

k = 20;
[zx,Tx,softx,llx] = emCluster(X,k,'k++');
f = figure;
plotClassify2D([],X,zx);
saveas(f,'emgm_20_kpp.png', 'png');