function Xext = fpoly(X, degree, useConstant)
% X=fpoly(X,degree,useConst) : create expanded polynomial features of up to given degree
%   useConstant=0 => do not include "constant" feature, x_0^(j) = 1 for all j

if (nargin < 3 ) useConstant = 1; end;
[m,n]=size(X);

if ((n+1)^(degree+1) > 1e7) 
  error(sprintf('%d^%d = too many potential output features?',n,degree)); 
end;


if (n==1)        % faster shortcut for scalar data
  if (useConstant) p=0:degree; else p=1:degree; end;
  Xext = repmat(X,[1, length(p)]) .^ repmat(p,[m,1]);
  return;
end;


K=0;                                             % First, count the terms we'll need:
for i=1:(n+1)^(degree+1),                        %   enumerate over all possible combinations of features
  alpha = cell(1,n);
  [alpha{:}] = ind2sub((degree+1)*ones(1,n),i);  %   convert index to exponents of each term
  alpha = [alpha{:}]-1; 
  if (sum(alpha)>degree) continue; end;          %   figure out if they're present in the degree-d polynomial
  %fprintf('['); fprintf('%d ',alpha); fprintf(']\n');  % print check for debugging
  K = K+1;                                       %   count how many terms there will be
end;

Xext = zeros(m,K);
k=1;                                             % Now, construct those terms:
for i=1:(n+1)^(degree+1),                        %   re-enumerate to compute term values
  alpha = cell(1,n);
  [alpha{:}] = ind2sub((degree+1)*ones(1,n),i);  %   convert index to exponents of each term
  alpha = [alpha{:}]-1;
  if (sum(alpha)>degree) continue; end;          %   figure out if they're present in the degree-d polynomial
  Xext(:,k) = prod(X.^repmat(alpha,[m,1]),2);
  k = k+1;
end;

if (~useConstant) Xext = Xext(:,2:end); end;


