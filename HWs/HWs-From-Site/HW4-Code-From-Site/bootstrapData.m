function [X Y] = bootstrapData(X,Y,nBoot)
% resample (bootstrap) data set
% [Xboot Yboot] = bootstrapData(X,Y,nBoot) resamples the data points (xi,yi) with replacement nBoot times
  [Nx,Mx] = size(X);
  idx = ceil(rand(1,nBoot)*Nx);
  X = X(idx,:);
  if (~isempty(Y))
    [Ny,My] = size(Y);
    if (Nx ~= Ny) error('X and Y should have the same number of rows'); end;
    Y = Y(idx,:);
  end;

