function [Yext values] = to1ofK(Y, values)
% [Ynew values] = to1ofK(Y [,values]) : convert discrete-valued Y into 1-of-K representation
%  optional "values" specifies / returns the possible values of Y

[N,D] = size(Y);
assert(D==1);   % need Y to be discrete scalar for this function

if (nargin < 2) values = unique(Y); end;
C = length(values); 

%index = toIndex(Y,values); 
index = zeros(size(Y));
for v=1:length(values),
  index(Y==values(v)) = v;
end;
if (any(index==0)) error('Target values Y contain a class not in allowed values list'); end;

Yext = zeros(N,C); 
Yext( (1:N) + N*(index'-1))=1;

%if (nargin < 2) values = unique(Y); end;
%[m,n] = size(Y);
%assert(n==1);   % need Y to be discrete scalar for this function
%
%Yext = zeros(m,length(values));
%for i=1:m,
%  %Yext(i, find(values==Y(i))) = 1;
%  Yext(i, :) = (values==Y(i))';
%end;

