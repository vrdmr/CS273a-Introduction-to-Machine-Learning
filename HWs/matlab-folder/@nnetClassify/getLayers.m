function S=getLayers(obj,sizes)
% getLayers(nnet) : get layer sizes of nnet

S = zeros(1,length(obj.wts)+1);
for l=1:length(S)-1, S(l)=size(obj.wts{l},2)-1; end;
S(end)=size(obj.wts{end},1);

