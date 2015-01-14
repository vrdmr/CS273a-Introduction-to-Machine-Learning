function [errorcount, errorrate] = errorTrainBayes(Yte, YteHat)
% Y = from1ofK(Y1k [,values]) : convert 1-of-K valued Y into discrete representation
%  optional "values" specifies the possible values of Y (default 1..K)

errorcount = 0;
for i=1:size(Yte,1);
    if(Yte(i) ~= YteHat(i));
        errorcount = errorcount + 1;
        [Yte(i) YteHat(i)]
    end;
end;

errorrate = errorcount/size(Yte,1);