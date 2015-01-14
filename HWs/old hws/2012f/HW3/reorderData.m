function [X Y] = reorderData(X,Y)
% reorder (permute) data set
% [X Y] = reorderData(X [,Y]) permutes the order of the data points (xi,yi) 
  [Nx,Mx] = size(X);
  pi = randperm(Nx);
  X = X(pi,:);
  if (nargin > 1 && ~isempty(Y))
    [Ny,My] = size(Y);
    if (Nx ~= Ny) error('X and Y should have the same number of rows'); end;
    Y = Y(pi,:);
  end;

