function [error] = errorTrain(Yte, YteHat)
% Y = from1ofK(Y1k [,values]) : convert 1-of-K valued Y into discrete representation
%  optional "values" specifies the possible values of Y (default 1..K)

error = 0;
for i=1:size(Yte,1);
    if(Yte(i) ~= YteHat(i));
        error = error + 1;
    end;
end;

error = error/size(Yte,1);

