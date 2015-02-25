function obj = train(obj, Xtr, Ytr, stepsize, tolerance, maxSteps, init)
% obj = train(obj, Xtrain, Ytrain, stepsize, tolerance, maxSteps, init)
%     Xtrain = [n x d] training data features (constant feature not included)
%     Ytrain = [n x 1] training data classes 
%     stepsize = step size for gradient descent (decreases as 1/iter)
%     tolerance = tolerance for stopping criterion
%     maxSteps = maximum number of steps before stopping
%     init = 'keep', 'zeros', 'randn'
%

if (nargin >= 7) obj=initWeights(obj,getLayers(obj),init,Xtr,Ytr);  end;  % init method
if (nargin < 6) maxSteps = 5000;  end;  % max number of iterations
if (nargin < 5) tolerance = 1e-4; end;  % stopping tolerance
if (nargin < 4) stepsize = .01;   end;  % gradient descent step size

plotFlag = 1;                    % with plotting

% Convert Ytrain to 1-of-K format
if (isempty(obj.classes)), [YtrK, obj.classes] = to1ofK(Ytr);
else                       [YtrK] = to1ofK(Ytr,obj.classes); 
end;

[n,d] = size(Xtr);            % d = dimension of data; n = number of training data
L = length(obj.wts)+1;           % get # of layers


% Define the desired activation function and its derivative for training
%Sig  = @(z) 1./(1+exp(-z));      % Logistic Sigmoid
%dSig = @(z) Sig(z).*(1-Sig(z));  % 
%Sig0 = Sig; dSig0 = dSig;        % Output layer (classifier => saturate output)
Sig=obj.Sig; dSig=obj.dSig; Sig0=obj.Sig0; dSig0=obj.dSig0;


% Outer loop of (stochastic) gradient descent:
iter=1;                                               % iteration #
done=0;                                               % end of loop flag
surr=zeros(1,maxSteps);                               % surrogate loss values
errs=zeros(1,maxSteps);                               % misclassification rate values
while (~done)
  % Step size evolution
  stepi = stepsize / iter;                            % classic 1/t decrease
  %stepi = stepsize;                                  % fixed stepsize


if (0) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Batch gradient update (one pass)
  for l=1:L-1, grad{l}=0*obj.wts{l}; end;                % init weight gradient entries to zero
  for i=1:n,
    [A,Z] = responses(obj.wts,Xtr(i,:),Sig,Sig0);     % compute all layers' responses, then backprop:
    delta = (Z{L} - YtrK(i,:)) .* dSig0(Z{L});        % Take derivative of output layer
    for l=L-1:-1:1,
      grad{l} = grad{l} + delta' * Z{l};                 % compute gradient on current layer wts
      delta = (delta*obj.wts{l}) .* dSig(Z{l});          % propagate gradient downward
      delta = delta(2:end);                              %   discard constant feature
    end;
  end;
  for l=1:L-1, obj.wts{l} = obj.wts{l} - stepi * grad{l}; end;   % take gradient step on weights

else %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Stochastic gradient update (one pass)
  for i=1:n,  % for each data example,
    [A,Z] = responses(obj.wts,Xtr(i,:),Sig,Sig0);     % compute all layers' responses, then backprop:
    delta = (Z{L} - YtrK(i,:)) .* dSig0(Z{L});        % Take derivative of output layer
    for l=L-1:-1:1,
      grad = delta' * Z{l};                              % compute gradient on current layer wts
      delta = (delta*obj.wts{l}) .* dSig(Z{l});          % propagate gradient downward
      delta = delta(2:end);                              %   discard constant feature
      obj.wts{l} = obj.wts{l} - stepi * grad;            % take gradient step on current layer weights
    end;
  end;

end; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Compute current error values
  errs(iter)  = errK(obj,Xtr,YtrK);                              % error rate (classification)
  surr(iter)  = mseK(obj,Xtr,YtrK);                              % surrogate (mse on output)
  %surr(iter)  = -loglikelihood(obj,Xtr,YtrK);                   % surrogate (neg likelihood)

  % Make plots, if desired
  if (plotFlag),
    fig(1); H=semilogx(1:iter, surr(1:iter),'b-',1:iter,errs(1:iter),'g-');
    switch d,             % Plots to help with visualization
      case 1, fig(2); plotRegress1D(obj,Xtr,Ytr); %  for 1D data, display the data and the function
      otherwise, % no plot for higher dimensions...     %  higher dimensions visualization is hard
    end;
    % Hack to plot internal (hidden) responses for debugging:
    %ax=axis; xs=linspace(ax(1),ax(2),100); [A,Z] = responses(obj.wts,xs',Sig,Sig0);
    %fig(3); plot(Z{1},'b-'); hold on; plot(Z{2},'r-'); plot(Z{3},'g-'); plot(Z{4},'k-'); hold off;
  drawnow;
  end;

  %for l=1:L-1, [obj.wts{l}, grad{l}], end;         % print weights and gradients (for debugging)

  % Various stopping conditions:
  %done = (iter >= maxSteps); 
  %done = (iter>1) && ( (abs(errs(iter)-errs(iter-1))<tolerance) || (iter >= maxSteps) );
  done = (iter>1) && ( (abs(surr(iter)-surr(iter-1))<tolerance) || (iter >= maxSteps) );
  %done = (abs(grad*grad')<tolerance) || (iter >= maxSteps);
  %done = (abs(wtsOld-obj.wts)<tolerance) || (iter >= maxSteps);
  iter = iter + 1;
  wtsOld = obj.wts;
end;


% get linear sum from previous layer (A) and saturated activation responses (Z) for a data point
function [A,Z] = responses(wts,Xin,Sig,Sig0)
  L = length(wts)+1;
  Z=cell(1,L); A=Z;
  A{1} = 1; Z{1} = [ones(size(Xin,1),1) Xin];% Compute linear combination of inputs
  for l=2:L-1
    A{l} = Z{l-1} * wts{l-1}';               % Compute linear combination of previous layer
    %Z{l} = [ones(size(A{l},1),1) 1./(1+exp(-A{l}))]; % Pass through activation f'n and add constant feature
    Z{l} = [ones(size(A{l},1),1) Sig(A{l})]; % Pass through activation f'n and add constant feature
  end;
  A{L} = Z{L-1} * wts{L-1}';
  Z{L} = Sig0(A{L});               % Output layer (saturate for classifier, not for regression)
  %Z{L} = 1./(1+exp(-A{L}));        % Classification (saturate outputs)
  %Z{L} = A{L};                    % Regression (don't saturate outputs)


function fig(i)
  if (ishandle(i)) set(0,'CurrentFigure',i);
  else figure(i); 
  end;
