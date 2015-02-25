function a = auc(obj,Xv,Yv);
% auc(obj,Xtest,Ytest) : compute the "area under the roc curve" score on test data
%  obj must be a binary classifier

if (length(obj.classes)>2) error('Currently only supports binary classification'); end;

try 
  % Compute "response" (soft binary classification score)
  soft = predictSoft(obj,Xv);
  soft = soft(:,2);            % Pr[class = 2nd]
catch 
  % Or, can use "hard" binary prediction if no soft prediction available
  soft = predict(obj,Xv);
end

% Sort data by score value
[srt,ord] = sort(soft'); Yv=Yv(ord)';
% Find ties in the sorting score
same = [srt(1:end-1)==srt(2:end) 0];
n=length(soft);
% Initialize rank assuming no ties
rnk = 1:n;

% Compute tied rank values
i=1; 
while (i<n)
  if (same(i))
    start = i;
    while (same(i)) i=i+1; end;
    rnk(start:i) = (i+start)/2;
  end;
  i=i+1;
end;

% # of true negatives and positives
n0 = sum(Yv==obj.classes(1));
n1 = sum(Yv==obj.classes(2));

if (n0==0 || n1 == 0) error('Data of both class values not found!'); end;

% compute AUC using Mann-Whitney U statistic
a = (sum(rnk(Yv==obj.classes(2))) - n1*(n1+1)/2)/n1/n0;

