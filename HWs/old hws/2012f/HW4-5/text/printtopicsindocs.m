%----------------------------------------------------
% file:    topicsindocs.m
% purpose: print topics from topic model run
% usage:   matlab> topicsindocs
% inputs:  Nwt.txt, vocab.txt
% outputs: topicsindocs.txt
% author:  David Newman, newman@uci.edu
%----------------------------------------------------

[did,tid,cnt] = textread('Ndt.txt','%d%d%d','headerlines',3);
Ndt = sparse(did,tid,cnt);
[D,T] = size(Ndt);

%alpha = 0.1;
alpha = 0.0;
Ndt = Ndt + alpha;
fac = 1 ./ sum(Ndt,2);
for t=1:T
  Ndt(:,t) = Ndt(:,t) .* fac;
end
prob_t_given_d = Ndt';

fid = fopen('topicsindocs2.txt','w');
[doc] = textread('docs.txt','%s');
assert(length(doc)==D)

for d=1:D
  prob = prob_t_given_d(:,d);
  [xsort,isort] = sort(-prob);
  fprintf(fid,'<%s> ', doc{d});
  for t=1:4
    tt = isort(t);
    if (prob(tt)>0.1)
      fprintf(fid,'t%d ', tt);
    end
  end
  fprintf(fid,'\n');
end

fclose(fid);
fprintf('wrote file topicsindocs2.txt\n');
