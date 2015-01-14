function [X Y] = shuffleData(X,Y,s)
% shuffle (permute) data set
% [X Y] = shuffleData(X [,Y, s]) permutes the order of the data points (xi,yi) 
% If s (seed) is given, it uses that, else uses random permutation.

  [Nx,Mx] = size(X);
  
  % Checking for the Seed.
  if(~isempty(s))
      pi = randperm(s, Nx);
  else
      pi = randperm(Nx);
  end;
  X = X(pi,:);
  
  if (nargin > 1 && ~isempty(Y))
    [Ny,My] = size(Y);
    if (Nx ~= Ny) error('X and Y should have the same number of rows'); end;
    Y = Y(pi,:);
  end;

