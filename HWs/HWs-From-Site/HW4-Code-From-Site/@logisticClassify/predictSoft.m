% perform "soft" prediction on Xtest (predicts real-valued #s)
% function YteSoft = predictSoft(obj,Xte)
function YteSoft = predictSoft(obj,Xte)
  YteSoft = logistic(obj,Xte);

