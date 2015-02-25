function h = fig(varargin)
% fig(...) : operates similarly to built-in "figure(...)", but avoids switching focus to window
% This is useful when plotting repeatedly, since it leaves the command window in focus
%
% Code by Rafael Oliveira, http://www.mathworks.com/matlabcentral/fileexchange/33987-let-me-work-figure

    if mod(numel(varargin),2)
        f = varargin{1};
        parameters = varargin(2:end);
    else
        parameters = varargin;
        f = [];
    end
    
    if ishandle(f)
        set(0,'CurrentFigure',f);
    elseif ~isempty(f)
        f = figure(f);
    else
        f = figure;
    end
    
    if ~isempty(parameters)
        set(f,parameters{:});
    end
    
    if nargout == 1
        h = f;
    end
end
