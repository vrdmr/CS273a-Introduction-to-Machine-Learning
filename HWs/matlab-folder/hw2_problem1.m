%% Problem 1
% Author: Varad Meru (vmeru@uci.edu)
% Course: CS 273, Machine Learning (http://sli.ics.uci.edu/Classes/2015W-273a)
% Homework Description - http://sli.ics.uci.edu/Classes/2015W-273a?action=download&upname=HW2.pdf
% Started on 16th Jab, 15.

% Fetching the dataset and separating it into X and Y.
curve=load('data/curve80.txt');     % load the text file
y = curve(:,end);           % target value is last column
X = curve(:,1);       % features are other columns

% Part (a)
% Splitting the data into 75-25 split.
[Xtr, Xte, Ytr, Yte] = splitData(X,y, .75);
whos

%% Part (b)
% Training the linear regression with Xtr as an input
lr = linearRegress( Xtr, Ytr );
xs = [0:.05:10]';
ys = predict(lr, xs);
ys1 = predict(lr, Xte);

% Plotting the training data and the predicted function in the same plot.
f=figure;
scatter(Xtr, Ytr, 'filled');
hold on;
plot(xs, ys);
hold off;
saveas(f,'lr.jpg','jpg');

% MSE for both Training and Test data
mse(lr, Xtr, Ytr)
% ans = 1.1277
mse(lr, Xte, Yte)
% ans = 2.2423

%% Part (c)

% XtrP = fpoly(Xtr, 1, false); % create poly features up to given degree; no "1" feature.
XtrP = Xtr;
[XtrP, M,S] = rescale(XtrP); % it's often a good idea to scale the features

lr = linearRegress( XtrP, Ytr ); % create and train model
XteP = rescale(Xte, M,S);
YteP = predict(lr, XteP);
mse(lr, XteP, YteP)


%% Some part
degree = 2;
Phi = @(x) rescale( fpoly(x,degree,false), M,S);
% XtrP = fpoly(Xtr, 2, false); % create poly features up to given degree; no "1" feature.
% [XtrP, M,S] = rescale(XtrP); % it's often a good idea to scale the features

lr = linearRegress(); % create and train model
YhatTrain = predict( lr, Phi(Xtr) );
YhatTest = predict( lr, Phi(Xte) );

%XteP = rescale(Xte, M,S);
%YteP = predict(lr, XteP);
%mse(lr, XteP, YteP)


%% Part (c-1)
% Training the linear regression with Xtr as an input
Xtr2 = [Xtr, Xtr.^2];
Ytr2 = [Ytr, Ytr.^2];
lr = linearRegress( Xtr2, Ytr2 );
xs = [0:.05:10]';
ys = predict( lr, xs );

% Plotting the training data and the predicted function in the same plot.
figure;
scatter(Xtr2, Ytr2);
hold on;
scatter(xs, ys);
hold off;

% MSE for both Training and Test data
%mse(lr, Xtr, Ytr);
% ans = 1.1277
%mse(lr, Xte, Yte);
% ans = 2.2423

%% Part (c - extended)
% Now converting into a polynomial function
% Squaring the elements of Xtr and using it to train the regression model.
%XtrP = fpoly(Xtr, 1, false); % create poly features up to given degree; no "1" feature.
%[XtrP, M,S] = rescale(XtrP); % it's often a good idea to scale the features 
%lr = linearRegress( XtrP, Ytr ); % create and train model
%ys = predict( lr, xs );

% now, apply the same polynomial expansion & scaling transformation to Xtest:
%XteP = rescale( fpoly(Xte,1,false), M,S);

% often we wish to apply some transformation (possibly data-dependent, like
% scaling) to the features. Ideally, we should then be able to apply this
% same transform to new, unseen test data when it arrives, so that it will
% be treated in exactly the same way as the training data. ?Feature
% transform? functions like rescale are written to output their settings,
% (here, M,S), so that they can be reused on subsequent data.

f = figure;
i = 1;
errorsTr = zeros(1, 6);
errorsTe = zeros(1, 6);
d=[1,3,5,7,10,18];
for degree=[1,3,5,7,10,18];
    
    XtrP = fpoly(Xtr, degree, false); % create poly features up to given degree; no "1" feature.
    [XtrP, M,S] = rescale(XtrP); % it's often a good idea to scale the features 
    lr = linearRegress( XtrP, Ytr ); % create and train model
    
    % XS vs YS plot. 
    xs = [0:.05:10]';
    xsP = rescale( fpoly(xs,degree,false), M,S);
    ysP = predict( lr, xsP );
    
    subplot(2,3,i);
    title(strcat({'degree= '},num2str(degree)))
    hold on;
    scatter(Xtr, Ytr, 'filled');
    ax = axis;
    hold on;
    plot(xs, ysP);
    axis(ax);
    
    
    XteP = rescale(fpoly(Xte,degree,false), M,S);
    YteP = predict(lr,XteP);
    
    % MSE for both Training and Test data
    errorsTr(1, i) = mse(lr, XtrP, Ytr)
    % ans = 1.1277
    errorsTe(1, i) = mse(lr, XteP, Yte)
    % ans = 2.2423
    i= i+1;
    % now, apply the same polynomial expansion & scaling transformation to Xtest:
    % 
end;
hold off;
saveas(f,'lr1.jpg','jpg');

f = figure;
semilogy(d, errorsTr);
hold on;
semilogy(d, errorsTe);
hold off;
legend('Training Error','Testing Error')
saveas(f,'lr2.jpg','jpg');

%%
Xtr2 = [Xtr, Xtr.^2];
Ytr2 = [Ytr, Ytr.^2];
Xte2 = [Xte, Xte.^2];

lr = linearRegress( Xtr2, Ytr );
YteP = predict(lr, Xte2);
plot(Xte2(:,1), Xte2(:,2));
hold on;
scatter(Xte2(:,1), Xte2(:,2));
hold on;
plot(YteP); hold off;


%% Problem ccccccc
% Author: Varad Meru (vmeru@uci.edu)
% Course: CS 273, Machine Learning (http://sli.ics.uci.edu/Classes/2015W-273a)
% Homework Description - http://sli.ics.uci.edu/Classes/2015W-273a?action=download&upname=HW2.pdf
% Started on 16th Jab, 15.

% Fetching the dataset and separating it into X and Y.
curve=load('data/curve80.txt');     % load the text file
y = curve(:,end);           % target value is last column
X = curve(:,1);       % features are other columns
whos

% Part (a)
% Splitting the data into 75-25 split.
[Xtr, Xte, Ytr, Yte] = splitData(X,y, .75);

degree = 2;
XtrP = fpoly(Xtr, degree, false); % create poly features up to given degree; no "1" feature.
[XtrP1, M,S] = rescale(XtrP); % it's often a good idea to scale the features 

XteP = rescale(fpoly(Xte, degree, false), M,S);
    
lr = linearRegress( XtrP1, Ytr ); % create and train model
YteP = predict( lr, XteP );
hold on;

lr = linearRegress(Xtr2,Ytr);

mse(lr, XteP, Yte)

temp = [XteP(:,1), YteP];

figure; scatter(XteP(:,1), YteP); hold on; scatter(XteP(:,1), Yte); hold off;

figure; plot(temp(:,1), temp(:,2)); hold on; scatter(XteP(:,1), Yte); hold off;


XteP = rescale( fpoly(Xte,degree,false), M,S);
    
    lr = linearRegress( XtrP, Ytr ); % create and train model
    YteP = predict( lr, XteP );
    