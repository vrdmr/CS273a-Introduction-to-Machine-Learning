function handle = plotGauss2D( gMean, gCov, colorString, varargin) 
% plot an ellipse indicating a Gaussian distribution using the given plot style string

if (length(gMean)>2) error('Can only plot 2-dimensional Gaussians'); end;

theta = [0:.01:2*pi]';
circle = [sin(theta), cos(theta)];
ell = circle * sqrtm(gCov);
ell = ell + ones(size(ell,1),1)*gMean;


handle = plot( gMean(1),gMean(2), [colorString 'x'], ell(:,1), ell(:,2), colorString , varargin{:});


