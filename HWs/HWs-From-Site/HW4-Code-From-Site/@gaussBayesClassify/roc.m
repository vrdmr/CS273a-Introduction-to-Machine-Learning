function [fpr,tpr,tnr] = roc(obj,Xv,Yv);
% [fpr,tpr,tnr] = roc(obj,Xtest,Ytest) : compute the "receiver operating characteristic" curve" on test data
%   obj must be a binary classifier
%   plot(fpr,tar,'-') to display the ROC curve
%   plot(tpr,tnr,'-') to display the Sensitivity/Specificity curve

if (length(obj.classes)>2) error('Currently only supports binary classification'); end;

try 
  % Compute "response" (soft binary classification score)
  soft = predictSoft(obj,Xv);
  soft = soft(:,2);            % Pr[class = 2nd]
catch 
  % Or, can use "hard" binary prediction if no soft prediction available
  soft = predict(obj,Xv);
end

% # of true negatives and positives
n0 = sum(Yv==obj.classes(1));
n1 = sum(Yv==obj.classes(2));

if (n0==0 || n1 == 0) error('Data of both class values not found!'); end;

% Sort data by score value
[srt,ord] = sort(soft'); Yv=Yv(ord)';

% compute false positive & true positive rates
tpr = cumsum(Yv(end:-1:1)==obj.classes(2)) ./ n1;  
fpr = cumsum(Yv(end:-1:1)==obj.classes(1)) ./ n0;  
tnr = cumsum(Yv==obj.classes(1))./n0; tnr = tnr(end:-1:1);

% Find ties in the sorting score
same = [srt(1:end-1)==srt(2:end) 0];

tpr = [0 tpr(~same)];
fpr = [0 fpr(~same)];
tnr = [1 tnr(~same)];