clear all; clc; close all;
addpath './Datasets'
addpath './Utility'

load ('YaleB_32x32');
fea = double(fea');
fea = fea./ repmat(sqrt(sum(fea.*fea)),[size(fea,1) 1]);
train_num = 25;

%------------------------------------------------
% Seperate train and test set
%------------------------------------------------
[trfea, trgnd,ttfea,ttgnd] = sperate_data(fea,gnd,train_num);

%------------------------------------------------
% first step, remove the mean!
%------------------------------------------------
XMean = mean(trfea,2);
trfea = trfea - repmat(XMean, 1, size(trfea,2));
YMean = mean(ttfea,2);
ttfea = ttfea - repmat(YMean, 1, size(ttfea,2));

%------------------------------------------------
% parameter settings
%------------------------------------------------
para.lambda   = 0.5; 
para.beta  = 0.05;
para.gamma = 0.05;
MaxIters   = 30;
knum       = 15;

%------------------------------------------------
% Main functions
%------------------------------------------------
% Supervised MSRL
[W, b, obj] = SMSRL(trfea, trfea, trgnd, para, MaxIters, knum);

% Semi-supervised MSRL
% full_fea = [trfea ttfea];
% [W, b, obj] = SMSRL(trfea, full_fea, trgnd, para, MaxIters, knum);

%------------------------------------------------
% Classification
%------------------------------------------------
usedTr = W'*trfea;%+b*ones(1,N)
usedTt = W'*ttfea;%+b*ones(1,N)
cclass = knnclassify(usedTt',usedTr',trgnd,1,'euclidean');
acc = 100*(sum(cclass==ttgnd))/numel(ttgnd);

disp(['--------------------------------------------------------']);
fprintf('Train_num = %d, lam=%.3f, beta=%.3f, gamma=%.3f, acc=%.2f...\r\n',train_num,para.lambda, para.beta, para.gamma, acc);

