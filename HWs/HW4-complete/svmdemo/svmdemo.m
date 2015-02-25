function model = svmdemo(X, y, kernel, C, gamma)
% SVMDEMO  Demos the SVM code
%   First generate some training data X, Y by using GENDATA(). Then try
%
%   SVMDEMO(X, Y, 'LINEAR', C) where C is the SVM C parameter.
%
%   SVMDEMO(X, Y, 'RBF', C, GAMMA) where C is the SVM C parameter and
%   GAMMA is the RBF GAMMA parameter
%
%   Restet the random number generator to run twice on the same
%   data. E.g.:
%
%     randn('state', 1) ;
%     [X,y] = gendata(15, 15) ;
%
%     figure(1) ;
%     svmdemo(X, y, 'linear', 1) ;
%
%     figure(2) ;
%     svmdemo(X, y, 'linear', .1) ;
%
%   See also:: SVM(), SVMDEMOALL().
%
%   Author:: Andrea Vedaldi <vedaldi@robots.ox.ac.uk>

% --------------------------------------------------------------------
%                                                       SVM parameters
% --------------------------------------------------------------------

if nargin < 1, kernel = 'linear' ; end
if nargin < 2, C = 1 ; end
if nargin < 3, gamma = 1 ; end

% --------------------------------------------------------------------
%                                                             Training
% --------------------------------------------------------------------

switch kernel
  case 'linear'
    K = X'*X ;
  case 'rbf'
    K = exp(- gamma * pdist2(X,X)) ;
end
model = svm(K,y,C) ;

% --------------------------------------------------------------------
%                                                              Testing
% --------------------------------------------------------------------

% evaluate the SVM f(x) on a dense grid

ur = linspace(-7,7,256) ;
[u,v] = meshgrid(ur) ;
X_dense = [u(:)' ; v(:)'] ;
switch kernel
  case 'linear'
    K_dense = X(:,model.svind)' * X_dense ;
  case 'rbf'
    K_dense = exp(- gamma * pdist2(X(:,model.svind), X_dense)) ;
end
f_dense = model.alphay(model.svind)' * K_dense + model.b ;
f_dense = reshape(f_dense, size(u,1),size(u,2)) ;

cla ;
imagesc(ur,ur,f_dense) ; colormap cool ; hold on ;
[c,hm] = contour(ur,ur,f_dense,[-1 -1]) ;
set(hm,'color', 'r', 'linestyle', '--') ;
[c,hp] = contour(ur,ur,f_dense,[+1 +1]) ;
set(hp,'color', 'g', 'linestyle', '--') ;
[c,hz] = contour(ur,ur,f_dense,[0 0]) ;
set(hz,'color', 'b', 'linewidth', 4) ;
hg  = plot(X(1,y>0), X(2,y>0), 'g.', 'markersize', 10) ;
hr  = plot(X(1,y<0), X(2,y<0), 'r.', 'markersize', 10) ;
hko = plot(X(1,model.svind), X(2,model.svind), 'ko', 'markersize', 5) ;
hkx = plot(X(1,model.bndind), X(2,model.bndind), 'kx', 'markersize', 5) ;
axis square ;
legend([hg hr hko hkx hz hp hm], ...
       'pos. vec.', 'neg. vec.', 'supp. vec.', 'margin vec.', ...
       'decision bound.', 'pos. margin', 'neg. margin', ...
       'location', 'northeastoutside') ;
switch kernel
  case 'linear'
    title(sprintf('linear kernel (C = %g)', C)) ;
  case 'rbf'
    title(sprintf('RBF kernel (C = %g, gamma = %g)', C, gamma)) ;
end
