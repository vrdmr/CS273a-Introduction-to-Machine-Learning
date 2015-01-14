    % Test function: predict on Xtest
    function Yte = predict(obj,Xte)
      Yte = Xte * obj.theta';
    end

