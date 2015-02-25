function obj=setActivation(obj,method,Sig,dSig,Sig0,dSig0)
% nnet = setActivation(obj, method, ...) : set activation function
%  method = {'logistic','htangent','custom'}
%  For 'custom', also pass activation functions: Sig(z), dSig(z), Sig0(z), dSig0(z)
%   e.g., Sig = @(z) 1./(1+exp(-z));  dSig(z) = Sig(z).*(1-Sig(z)) for logistic
%   Sig0 and dSig0 are the output layer activation functions

% TODO : add multilogit?

switch(lower(method))
case 'logistic',
  Sig   = @(z) 2*1./(1+exp(-z)) - 1;
  dSig  = @(z) 2*exp(-z)./(1+exp(-z)).^2;
  %dSig  = @(z) Sig(z).*(1-Sig(z));
  if isa(obj,'nnetClassify') Sig0  = Sig; dSig0 = dSig;      % for classification
  else                       Sig0  = @(z) z; dSig0 = @(z) 1+0*z; % for regression
  end;
case 'htangent',
  Sig   = @(z) tanh(z);
  dSig  = @(z) 1-tanh(z).^2;
  %Sig   = @(z) (1-exp(-2*z))./(1+exp(-2*z));
  %dSig  = @(z) 2*(1+Sig(z)).*exp(-2*z)./(1+exp(-2*z));
  if isa(obj,'nnetClassify') Sig0  = Sig; dSig0 = dSig;      % for classification
  else                       Sig0  = @(z) z; dSig0 = @(z) 1+0*z; % for regression
  end;
case 'custom',
end;

obj.activation = method;
obj.Sig   =  Sig;
obj.dSig  = dSig;
obj.Sig0  =  Sig0;  
obj.dSig0 = dSig0;  
