function [X,R] = imputeMissing(X,method,R);
% [X,R] = imputeMissing(X,method [,R]) -- impute missing values of features X in one of several simple ways
%   method = 'mean'     : fill missing values (nan) with mean of that feature over X
%            'median'   : as mean, but use median value
%            'gaussian' : fill with conditional mean assuming a Gaussian model on X (w shrinkage to N(0,1))
%            'constant' : replace missing values (nan) with constant value R

[N,F] = size(X);  % N data, F features

if (nargin < 3)     % Parameters R not given, compute from data
  switch (lower(method)),

  case 'median',
    R = zeros(1,F);
    for i=1:F,
      R(i) = median(X( ~isnan(X(:,i)), i));
      if (isnan(R(i))) R(i)=0; end;
    end;
  
  case 'mean',
    R = zeros(1,F);
    for i=1:F,
      R(i) = mean(X( ~isnan(X(:,i)), i));
      if (isnan(R(i))) R(i)=0; end;
    end;
  
  
  case 'gaussian',
    % compute gaussian model of features to infer missing values
    % to ensure invertable, use F (= # dimensions) pseudocounts of independent unit noise
    R={ zeros(1,F), zeros(F,F) };
    for i=1:F,
      R{1}(i) = mean(X( ~isnan(X(:,i)), i));
      if (isnan(R{1}(i))) R{1}(i)=0; end;
      ni = sum(~isnan(X(:,i)));
      R{1}(i) = R{1}(i) * (ni/(ni+F));        % shrink toward zero by F counts
    end;
    for i=1:F, for j=i:F,
      nans = isnan(X(:,i)) | isnan(X(:,j));
      nij = sum(~nans);
      R{2}(i,j) = mean( (X(~nans,i)-R{1}(i)).*(X(~nans,j)-R{1}(j)) );
      if (isnan(R{2}(i,j))) R{2}(i,j)=0; end;
      if (i==j) R{2}(i,j) = R{2}(i,j) * (nij/(nij+F)) + (F/(nij+F)); 
      else      R{2}(i,j) = R{2}(i,j) * (nij/(nij+F));  % shrink towards identity covar by F counts
      end;
      R{2}(j,i) = R{2}(i,j);                  % fill lower diagonal
    end; end;
  
  end;
end;

switch (lower(method)),
  case 'constant', 
    X( isnan(X(:)) ) = R; 
  case {'median','mean'}
    for i=1:F, X( isnan(X(:,i)),i )=R(i); end;
  case 'gaussian',
    mu = R{1}; Sig=R{2};
    for j=1:N,
      nans = isnan(X(j,:));    % find missing and non-missing for data point j & compute posterior mean
      X(j,nans) = mu(:,nans) - (Sig(nans,~nans) * inv(Sig(~nans,~nans)) * (X(j,~nans)-mu(~nans))')';
    end;
  otherwise, error('Unknown imputation method');
end;
