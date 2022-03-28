function [trfea, trgnd,ttfea,ttgnd] = sperate_data(fea,gnd,train_num)

if nargin < 3
   error('Error, please give the number of training samples per class!!!')
end;

rand('state',40);
class_num = length(unique(gnd));
num_class = zeros(class_num,1);
for k=1:class_num
    num_class(k,1) = length(find(gnd==k));
end

trfea = []; ttfea = [];
trgnd = []; ttgnd = [];
for j = 1:class_num
    index = find(gnd == j); 
    randIndex = randperm(num_class(j));
    trfea = [trfea fea(:,index(randIndex(1:train_num)))];
    trgnd = [trgnd ; gnd(index(randIndex(1:train_num)))];
    ttfea = [ttfea fea(:,index(randIndex(train_num+1:end)))];
    ttgnd = [ttgnd ; gnd(index(randIndex(train_num+1:end)))];
end
end