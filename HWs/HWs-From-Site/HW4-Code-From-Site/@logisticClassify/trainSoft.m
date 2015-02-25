function obj = train(obj, Xtrain, Ytrain, stepsize, tolerance, maxSteps, init, plotFlag)
% obj = logistic_train(obj, Xtrain, stepsize, tolerance, maxSteps, init, plotFlag)
%     Xtrain = [n x d] training data features (constant feature not included)
%     Ytrain = [n x 1] training data classes (0 or 1)
%     stepsize = step size for gradient descent
%     tolerance = tolerance for stopping criterion
%     maxSteps = maximum number of steps before stopping
%     init = 0: init to all zeros;  1: init to random weights;  
%     plotFlag = 0/1 : 0=no plots, 1=plot 1D prediction against X(1,:); 2=plot 2D decision boundary
% wts = [1 x d+1] vector of weights; wts(1) + wts(2)*X(:,1) + wts(3)*X(:,2) + ...
%
if (nargin < 8) plotFlag = 0;     end;  % set default values for options...
if (nargin < 7) init = 0;         end;  % initialization method
if (nargin < 6) maxSteps = 5000;  end;  % max number of iterations
if (nargin < 5) tolerance = 1e-4; end;  % stopping tolerance
if (nargin < 4) stepsize = .01;   end;  % gradient descent step size

[n,d] = size(Xtrain);            % d = dimension of data; n = number of training data
Xtrain1= [ones(n,1), Xtrain];    % make a version of training data with the constant feature

% SOFT training: assumes already in canonical form, but possibly real-valued, y=[0,1]
%obj.classes = unique(Ytrain);    % get class values in input
obj.classes=[0 1];
%if (length(obj.classes)~=2) error('Logistic regression should have two classes'); end;
%Ytrain( Ytrain==obj.classes(1) )=0;  % and convert to canonical 0/1 class values
%Ytrain( Ytrain==obj.classes(2) )=1;

switch init,
  case 0, obj.wts = zeros(1,d+1);     % init to all-zero
  case 1, obj.wts = randn(1,d+1);     % init at random
end;
wtsold = 0*obj.wts+inf;

iter=1; mse=zeros(1,maxSteps); err=zeros(1,maxSteps); done=0;
while (~done) 
  Yhat = logistic(obj, Xtrain);       % compute outputs: sigma( Xtrain * wts' )
  mse(iter) = mean( (Ytrain-Yhat).^2 );
  err(iter) = mean( (Ytrain ~= (Yhat > .5) ) );
  if (plotFlag), switch d,           % Plots to help with visualization
    case 1, plot1DLogistic(obj,Xtrain,Ytrain);    %  for 1D data we can display the data and the function
    case 2, plot2DLogistic(obj,Xtrain,Ytrain);    %  for 2D data, just the data and decision boundary
    otherwise, % no plot for higher dimensions... %  higher dimensions visualization is hard
  end; end;
  figure(1); plot(1:iter, mse(1:iter),'b-',1:iter,err(1:iter),'g-');
  for i=1:n,
    yhati = logistic(obj,Xtrain(i,:));        % compute output for example i
    grad = (yhati-Ytrain(i,:)) * yhati*(1-yhati) * Xtrain1(i,:);
    obj.wts = obj.wts - stepsize/iter * grad;      % take a step down the gradient
  end;
  if (iter>1) done = (abs(mse(iter)-mse(iter-1))<tolerance) || (iter > maxSteps); end;
  %done = (abs(grad*grad')<tolerance) || (iter > maxSteps);
  %done = (iter > maxSteps);
  iter = iter + 1;
end;




