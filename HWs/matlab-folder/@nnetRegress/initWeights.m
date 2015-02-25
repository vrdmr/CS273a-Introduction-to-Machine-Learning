function obj=initWeights(obj,sizes,method,X,Y)
% nnet = initWeights(obj, S,method,X,Y) : initialize weights of the NN
%   set layer sizes to S=[Ninput, N1, N2, ... Noutput] 
%   and set using "fast" method ('none','zeros','random')

switch (lower(method))
case 'none',
  % "no init" = do nothing
case 'zeros',
  for l = 1:length(sizes)-1, obj.wts{l} = zeros(sizes(l+1),sizes(l)+1); end;
case 'random',
  for l = 1:length(sizes)-1, 
    obj.wts{l} = .25*randn(sizes(l+1),sizes(l)+1);
  end;
case 'autoenc',
  % !!! TODO
  % 
case 'regress',
  % !!! TODO
  % 
otherwise,
  error(['Unknown initialization method ' method]);
end;

