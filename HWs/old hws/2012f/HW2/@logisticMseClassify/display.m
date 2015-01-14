    % display function, print out coefficients
    function display(obj)
      fprintf('Linear Binary Classifier; %d features\n',length(obj.wts));
      disp(obj.wts);
    end

