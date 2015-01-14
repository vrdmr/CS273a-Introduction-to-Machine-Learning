function [Y] = from1ofK(Yext, values)
% Y = from1ofK(Y1k [,values]) : convert 1-of-K valued Y into discrete representation
%  optional "values" specifies the possible values of Y (default 1..K)

[nil, Y] = max(Yext,[],2);
if (nargin > 1) Y=values(Y); end;

