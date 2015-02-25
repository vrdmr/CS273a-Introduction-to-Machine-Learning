function plotPairs(X,Y)
% plotPairs(X [,Y]) : create pair plot of features X, optionally colored by discrete class Y

[nData nFeat] = size(X);
if (nargin<2) Y = ones(nData,1); end;

cmap = jet(256);
clim = unique(Y)';
if (length(clim)==1) col = 1; 
else col = fix((clim - min(clim))./(max(clim)-min(clim)).*255)+1;
end;

clf;
for i=1:nFeat,
  for j=1:nFeat,
    subplot(nFeat,nFeat,i+(j-1)*nFeat);
    if (i==j) histy(X(:,i),Y);
    else
      hold on;
      for c=1:length(clim), 
        idx=find(Y==clim(c)); 
        %plot(X(idx,i),X(idx,j),'.','color',cmap(col(c),:),'markerfacecolor',cmap(col(c),:));
        plot(X(idx,i),X(idx,j),'o','color',cmap(col(c),:),'markerfacecolor',cmap(col(c),:));
      end;
      hold off;
    end;
  end;
end; 


