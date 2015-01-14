%----------------------------------------------------
% file:    topics.m
% purpose: print topics from topic model run
% usage:   matlab> topics
% inputs:  Nwt.txt, vocab.txt
% outputs: topics.txt
% author:  David Newman, newman@uci.edu
%----------------------------------------------------

[wid,tid,cnt] = textread('Nwt.txt','%d%d%d','headerlines',3);
Nwt = sparse(wid,tid,cnt);
[W,T] = size(Nwt);

beta = 0.01;
Nwt = Nwt + beta;

fac = 1 ./ sum(Nwt,1);
prob_w_given_t = Nwt * diag(fac);
  
prob_t = sum(Nwt,1);
prob_t = prob_t / sum(prob_t);

fid = fopen('topics2.txt','w');
[word] = textread('vocab.txt','%s');
assert(length(word)==W)

[xxsort,iisort] = sort(-prob_t);

for tt=1:T
  %t = iisort(tt);
  t = tt;
  prob = prob_w_given_t(:,t);
  [xsort,isort] = sort(-prob);
  %fprintf(fid,'[t%d] (prob=%.3f) ', t, prob_t(t));
  fprintf(fid,'[t%d] ', t);
  for w=1:12
    fprintf(fid,'%s ', word{isort(w)});
  end
  fprintf(fid,'...\n');
end

fclose(fid);
fprintf('wrote file topics2.txt\n');
