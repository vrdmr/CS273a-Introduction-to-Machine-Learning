function h = histy(X,Y,options);
% compute multi-class histograms and plot nicely

yvals = unique(Y);

if (nargin < 3) [nil,centers] = hist(X);                        % make consistent bin centers across classes
elseif (length(options)==1) [nil,centers] = hist(X,options);   %  << if # bins is specified
else centers = options;
end;
delta = [centers(2:end)-centers(1:end-1)]/2; delta(end+1)=delta(end);
width = .25 + .75/(1+log10(length(yvals)));

if (length(yvals)==1) cidx = 1;
else cidx = fix((yvals - min(yvals))./(max(yvals)-min(yvals)).*255)+1;
end;
Colors = jet(256); Colors=Colors(cidx,:);

 
H = zeros(length(yvals),length(centers));

for c=1:length(yvals),
 H(c,:) = hist(X(Y==yvals(c),:),centers);
end;

% sort each bin, keeping indices/permutations
[S,C] = sort(H,1,'descend');

holds = ishold;
% run through bins k max-to-min X colors c, plotting values of (k && index = c)
for k=1:size(H,1),
  for c=1:size(H,1),
    h = S(k,:) .* (C(k,:)==c);
    %hand = bar(centers,h,1); set(hand,'facecolor',Colors(c,:),'edgecolor','w');
    hand = bar(centers+(c-1)/size(H,1)*delta,h,width); set(hand,'facecolor',Colors(c,:),'edgecolor','w');
    %hand = bar(centers+(c-1)/size(H,1)*delta,h,width); set(hand,'facecolor',Colors(c,:),'edgecolor',Colors(c,:));
    hold on;
  end;
end;
if (~holds) hold off; end;

% Alternate plotting method (bars next to each other instead of on top)
%figure; bar(centers,H');
%

