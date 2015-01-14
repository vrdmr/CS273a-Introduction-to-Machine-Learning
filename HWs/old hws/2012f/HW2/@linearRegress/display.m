    % display function, print out coefficients
    function display(obj)
      fprintf('Linear Regression Object; %d features\n',length(obj.theta));
      disp(obj.theta);
    end

