%% Problem 1
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
whos
%%

i = 1;
nFolds = 5;
d=[1,3,5,7,10,18];
for degree=[1,3,5,7,10,18];
    % Degrees and scaling of the data
    XtrP = fpoly(Xtr, degree, false); % create poly features up to given degree; no "1" feature.
    [XtrP, M,S] = rescale(XtrP); % it's often a good idea to scale the features 
    
    XteP = rescale(fpoly(Xte,degree,false), M,S);
    
    for iFold = 1:nFolds;
        [Xti,Xvi,Yti,Yvi] = crossValidate(XtrP,Ytr,nFolds,iFold);
        % take ith data block as v learner = linearRegress(... 
        lr = linearRegress( Xti, Yti ); % create and train model
        % TODO: train on Xti, Yti , the data for this fold J(iFold) = ... 
        
        % TODO: now compute the MSE on Xvi, Yvi and save it 
        J(iFold) = mse(lr, Xvi, Yvi);
    end;
    % the overall estimated validation performance is the average of the performance on ea
    errors(i) = mean(J);
    lr = linearRegress( XtrP, Ytr ); % create and train model
    errorsTe(i) = mse(lr, XteP, Yte);
    i = i + 1;
end;

f = figure;
semilogy(d, errors)
legend('Training Error')
saveas(f,'cv.jpg','jpg');

f = figure;
semilogy(d, errors)
hold on;
semilogy(d, errorsTe);
hold off;
legend('Training Error','Testing Error')
saveas(f,'cv1.jpg','jpg');
%%
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