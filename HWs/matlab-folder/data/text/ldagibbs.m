clear all

% This code is originally by David Newman
% Modified by Alexander Ihler 2011
%


%--------------------------------------------
% parameters
%--------------------------------------------
T     = 8;     % number of topics
beta  = 0.01;  % Dirichlet prior on words
alpha = 0.1;   % Dirichlet prior on topics
rand('state',sum(100*clock));


%--------------------------------------------
% read corpus of documents
%--------------------------------------------
[did,wid,cnt] = textread('docword.txt','%d%d%d','headerlines',3);
X = sparse(did,wid,cnt);
D = max(did);       % number of docs
W = max(wid);       % size of vocab
N = sum(cnt);       % total number of words
[word] = textread('vocab.txt','%s');
assert(length(word)==W)


%--------------------------------------------
% allocate memory
%--------------------------------------------
w = zeros(N,1);    % represents word id (e.g. "ability") of the ith token in the corpus
d = zeros(N,1);    % ""         document id  (1..D) of ""
z = zeros(N,1);    % current value of token i's assigned topic
wp = zeros(W,T);   % total times word w has been assigned to topic t
dp = zeros(D,T);   % total times a token in document d has been assigned to topic t


%--------------------------------------------
% fill w and d
%--------------------------------------------
count = 1;
for j = 1:length(cnt)            % for each word in each document
  for i = count:count+cnt(j)-1   % we saw that word cnt(j) many times
    w(i) = wid(j);               % so store its word ID 
    d(i) = did(j);               % and document ID
  end
  count = count + cnt(j);        % keep track of how many total tokens we have seen
end
assert(max(w)==max(wid))
assert(max(d)==max(did))
assert(count-1==N)


%--------------------------------------------
% random initial assignment
%--------------------------------------------
z = floor(T*rand(N,1)) + 1;      % assign every token to a topic at random
for n = 1:N
  wp(w(n),z(n)) = wp(w(n),z(n)) + 1;   % count up number for each word ID
  dp(d(n),z(n)) = dp(d(n),z(n)) + 1;   % and each document ID
end
ztot    = sum(wp,1);                   % this should be the total number assigned to topic t
ztotchk = sum(dp,1);                   % calculate it in 2 ways and check that they are the same
assert(norm(ztot-ztotchk)==0)
assert(sum(ztot)==N)

%--------------------------------------------
% gibbs sampler
%   Sample a new value for each token's topic given all the other tokens' topics
%--------------------------------------------
for iter = 1:200
tic,
  for i = 1:N
  
    t = z(i);
    wp(w(i),t) = wp(w(i),t) - 1;   % remove token i's assignment from our count vectors
    dp(d(i),t) = dp(d(i),t) - 1;
    ztot(t)    = ztot(t)    - 1;

		%%% original version:          % Compute topic probability distribution given current counts
    %for t = 1:T                   % 
    %  probs(t) = (wp(w(i),t) + beta)/(ztot(t) + W*beta) * (dp(d(i),t) + alpha);
    %end
		%%% vectorized version:
    probs = (wp(w(i),:) + beta)./(ztot + W*beta) .* (dp(d(i),:) + alpha);

    %probs = probs/sum(probs);     % normalize probability distribution and draw a random sample
    %cumprobs = cumsum(probs);
		cumprobs = cumsum(probs); cumprobs=cumprobs/cumprobs(end);
    t = find(cumprobs>rand,1);

    z(i) = t;                      % now add this assignment back into our count vectors
    wp(w(i),t) = wp(w(i),t) + 1;
    dp(d(i),t) = dp(d(i),t) + 1;
    ztot(t)    = ztot(t)    + 1;
    
  end
  
  fprintf('iter %d (%f sec)\n', iter,toc);
  
  %--------------------------------------------
  % print current topics  (most likely words in each)
  %--------------------------------------------
  for t = 1:T
    fprintf('\t[%d] (%.3f) ', t, ztot(t)/N);    % print topic ID # and fraction of tokens it explains
    [xsort,isort] = sort(-wp(:,t));             % find the most likely words for topic t
    for ww = 1:8                                % in decreasing order, print each word
      fprintf('%s ', word{ isort(ww) } );
    end
    fprintf('\n');
  end

end
