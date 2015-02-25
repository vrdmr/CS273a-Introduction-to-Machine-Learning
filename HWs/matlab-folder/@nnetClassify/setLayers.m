function obj=setLayers(obj,sizes)
% nnet = setLayers(obj, S) : set layer sizes to S=[Ninput, N1, N2, ... Noutput]

  for l = 1:length(sizes)-1, 
    obj.wts{l} = randn(sizes(l+1),sizes(l)+1);     % initialize weight matrices
  end;  

