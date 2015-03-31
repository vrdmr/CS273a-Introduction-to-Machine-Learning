%% Homework 5 - Problem 2: k-means on text
clc;close all;clear;
[vocab] = textread('data/text/vocab.txt','%s');
[did,wid,cnt] = textread('data/text/docword.txt','%d%d%d','headerlines',3);

X = sparse(did,wid,cnt); % convert to a matlab sparse matrix
D = max(did);            % number of docs
W = max(wid);            % size of vocab
N = sum(cnt);            % total number of words

% It is often helpful to normalize by the document length:
Xn= X ./ repmat(sum(X,2),[1,W]) ; % divide word counts by doc length


%% Problem (a, b)
k = 20;
[z,c,sumd] = kmeans(Xn,k);
disp(sumd)
disp('--------------')
for i=1:20;
    [z1,c1,sumd1] = kmeans(Xn,k);
    disp(sumd1)
    if sumd1 < sumd
        z = z1;
        c = c1;
        sumd = sumd1;
    end;
end;
display('Minimum-sumd')
disp(sumd)
%% Problem (c) - 1
[uniques,numUnique] = count_unique(z);
[uniques,numUnique]

%{
ans =
     1    21
     2    45
     3     4
     4     2
     5     7
     6     8
     7    69
     8     3
     9    10
    10     2
    11     2
    12     1
    13     1
    14     3
    15     1
    16     1
    17    15
    18     2
    19     3
    20     2
%}

f=figure;
bar(uniques, numUnique);
saveas(f,'bar1.png','png')

%% Problem (c) - 2
for i =1:size(c, 1);
    [sorted,order] = sort( c(i,:), 2, 'descend');
    fprintf('Doc %d: ',i);
    fprintf('%s ', vocab{order(1:10)});
    fprintf('\n');
end;
%{
Doc 1: times square millennium city 2000 night 000 eve york midnight 
Doc 2: team game season coach games players league play going win 
Doc 3: archbishop york bishop cardinal sports church began column american close 
Doc 4: america boat team zealand cup nippon gilmour challengers round true 
Doc 5: book century marks amp war week finds lives school boy 
Doc 6: yeltsin putin russia russian president power political kremlin chechnya russians 
Doc 7: city american national president 000 home millennium end political going 
Doc 8: fireworks island city midnight celebration lot millennium hour calls celebrations 
Doc 9: y2k koskinen system problems saturday 2000 reported computers friday officials 
Doc 10: tutsi hutu rwanda burundi ethnic country experts africa van 1994 
Doc 11: texas arkansas yards line offensive game season sacks games defensive 
Doc 12: test end houston 000 0101 0102 100 1900 1900s 1968 
Doc 13: cats beijing owners police association called carry chinese eat eating 
Doc 14: hijackers hostages pakistan burger told government indian india passengers killed 
Doc 15: sports angeles began brooklyn column los seen young eye game 
Doc 16: economy government putin system america businesses country economic president russia 
Doc 17: 2000 computer internet government systems york problem problems news city 
Doc 18: lakers jackson game star phil players record conference practice monday 
Doc 19: news atlanta constitution journal service moved cox y2k cnn millennium 
Doc 20: buses authority diesel natural gas plan mta city york hybrid 
%}

%% Problem (d)

for i = [1, 15, 30];
    c(z(i,1),:)
end;

%% Debugging
euc = @(X, Y) norm(X - Y);
sqrd_sum = zeros(size(c,1),1);

for i = 1 : size(Xn, 1);
    cluster_num = z(i,:);
    disp(cluster_num);
    val = euc(Xn(i,:), c(z(i,:),:));
    sqrd_sum(cluster_num,1) = sqrd_sum(cluster_num,1) + val; 
end;

for i =1:size(Xn, 1);
    [sorted,order] = sort( Xn(i,:), 2, 'descend');
    fprintf('Doc %d: ',i);
    fprintf('%s ', vocab{order(1:10)});
    fprintf('\n');
end;
