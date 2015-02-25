function D2 = pdist2(X,Y)
% PDIST2  All pairwise squared Euclidean distances
%   D2 = PDIST2(X,Y) computes a matrix D2 such that
%
%    D2(i,j) = || X(:,i) - Y(:,j) ||^2
%
%   where || v || is the Euclidean norm of the vector v.
%
%   Author:: Andrea Vedaldi <vedaldi@robots.ox.ac.uk>

M = size(X,2) ;
N = size(Y,2) ;
D = size(X,1) ;

D2 = zeros(M,N) ;
for d = 1:D
  D2 = D2 + (X(d,:)'*ones(1,N) - ones(M,1)*Y(d,:)).^2 ;
end
