function [Yext values] = toIndex(Y, values)
% [Y2 values] = toIndex(Y [,values]) : convert discrete-valued Y into {1..K} representation
%  optional "values" specifies / returns the possible values of Y

if (nargin < 2) values = unique(Y); end;
[m,n] = size(Y);
assert(n==1);   % need Y to be discrete scalar for this function

Yext = zeros(m,1);
for i=1:length(values)
  Yext(find(Y==values(i))) = i;
end;

