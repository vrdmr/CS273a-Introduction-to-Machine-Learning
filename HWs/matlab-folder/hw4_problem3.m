%% Homework 4 - Problem 3 
% Author: Varad Meru (vmeru@uci.edu)
% Course: CS 273, Machine Learning (http://sli.ics.uci.edu/Classes/2015W-273a)
% Homework Description - http://sli.ics.uci.edu/Classes/2015W-273a?action=download&upname=HW3.pdf
% Started on 26th Jab, 15.

% Problem a - Decision trees on Kaggle

weatherX=load('data/kaggle/kaggle.X1.train.txt');     % load the text file
weatherY=load('data/kaggle/kaggle.Y.train.txt');     % load the text file
% feature_names=load('data/kaggle/kaggle.X1.names.txt');
X = weatherX;
Y = weatherY;

[X, Y] = shuffleData(X,Y);
[Xtr, Xte, Ytr, Yte] = splitData(X,Y, .75); % split data into 75/25 train/test

dt = treeRegress(Xtr,Ytr, 'maxDepth',20);
mse(dt,Xte,Yte) 
% ans = 0.7367

dt = treeRegress(Xtr,Ytr, 'maxDepth',15);
mse(dt,Xte,Yte)
% ans = 0.6384

%% Part b
for f=1:20;
    dt = treeRegress(Xtr,Ytr, 'maxDepth',f);
    errorsTe(f) = mse(dt,Xte,Yte);
    errorsTr(f) = mse(dt,Xtr,Ytr);
end;

K = (1:20);
h=figure;
semilogx(log(K), errorsTr);
hold on;
semilogx(log(K), errorsTe);
saveas(h,'maxdepth.jpg','jpg');


%{
mseYte =
  Columns 1 through 11
    0.5747    0.5205    0.4863    0.4704    0.4601    0.4488    0.4390    0.4429    0.4534    0.4721    0.5050
  Columns 12 through 20
    0.5349    0.5775    0.6109    0.6365    0.6521    0.6885    0.6998    0.7209    0.7335
%}

%% Part c
mxDepth = 20;
for f=3:12;
    dt = treeRegress(Xtr,Ytr, 'maxDepth',mxDepth,'minParent',2^f);
    errorsTe1(f) = mse(dt,Xte,Yte);
    errorsTr1(f) = mse(dt,Xtr,Ytr);
end;

K = [3:12];
h=figure;
semilogx(log(K), errorsTr1(:,3:12));
hold on;
semilogx(log(K), errorsTe1(:,3:12));
saveas(h,'minParent.jpg','jpg');

errorsTe
min(errorsTe)
%% Part c
mxDepth = 20;
smallestD = 0;
smallestP = 0;
smallestMSE = 100000;
i = 1;

for d=08:09;
    for p=[350 360 370 380 390 400];
    %for p=[1 10 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1024];
        dt = treeRegress(Xtr,Ytr, 'maxDepth',d,'minParent',p);
        temp = mse(dt,Xte,Yte);
        disp([temp d p]);
        if temp < smallestMSE
            errors(i,:) = [temp d p];
            disp(errors(i,:));
            smallestMSE = temp;
            smallestD = d;
            smallestP = p;
            i = i+1;
        end
    end;
end;

%{
errors

errors =

    0.5747    1.0000    3.0000
    0.5205    2.0000    3.0000
    0.4863    3.0000    3.0000
    0.4704    4.0000    3.0000
    0.4703    4.0000   10.0000
    0.4601    5.0000    3.0000
    0.4488    6.0000    3.0000
    0.4488    6.0000    5.0000
    0.4383    7.0000    3.0000
    0.4383    7.0000    7.0000
    0.4375    8.0000    8.0000
%}
% Lowest 8,8

%% Kaggle Submission
weatherX=load('data/kaggle/kaggle.X1.train.txt');     % load the text file
weatherY=load('data/kaggle/kaggle.Y.train.txt');     % load the text file
% feature_names=load('data/kaggle/kaggle.X1.names.txt');
X = weatherX;
Y = weatherY;
Xeval = load('data/kaggle/kaggle.X1.test.txt');
dt = treeRegress(X,Y, 'maxDepth',11,'minParent',2^6);
Ye = predict( dt, Xeval );     % make predictions
fh = fopen('predictions.csv','w');  % open file for upload
fprintf(fh,'ID,Prediction\n');      % output header line
for i=1:length(Ye),
    fprintf(fh,'%d,%d\n',i,Ye(i));  % output each prediction
end;
fclose(fh);                         % close the file