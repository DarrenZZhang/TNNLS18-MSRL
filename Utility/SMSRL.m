function [W, b, obj] = SMSRL(X, fea, class_id, para, iters, knum, r)
% X:              each column is a data point
% class_id:     a column vector, a column vector,  such as  [1, 2, 3, 4, 1, 3, 2, ...]'
% iters:          iteration times
% epsilon:      convergence
if nargin < 7
    r = -1;
end;
if nargin < 6
    knum = 15;
end;
if nargin < 5
    iters = 30;
end;

[dim, N] = size(X);
[~, tolN] = size(fea);
% num_class = max(class_id);
num_class = numel(unique(class_id));
% c = para.c;
c = min(5*max(floor((num_class-1)/5),1),num_class-1);
lambda = para.lambda;
beta   = para.beta;
gamma  = para.gamma;

distX = L2_distance_1(fea,fea);
[distX1, idx] = sort(distX,2);
A = zeros(tolN);  
rr = zeros(tolN,1);
for i = 1:tolN
    di = distX1(i,2:knum+2);
    rr(i) = 0.5*(knum*di(knum+1)-sum(di(1:knum)));   
    id = idx(i,2:knum+2);
    A(i,id) = (di(knum+1)-di)/(knum*di(knum+1)-sum(di(1:knum))+eps);
end;
A0 = (A+A')/2;
D0 = diag(sum(A0));
L = D0 - A0;
clear A0 D0;

if r <= 0
    r = mean(rr);
end;

T = zeros(num_class,N);
for i = 1 : N
    T(class_id(i),i)  = 1.0;
end

[W, ~] = LSR(X,  T,  beta);  % Here we use the soultion to the standard least squares regreesion as the initial solution

XMean = mean(X,2);
X = X - repmat(XMean, 1, N);
Q = eye(dim,c);
obj = zeros(iters,1);
for iter = 1:iters
    % Optimize Q P
    P = Q'*W;
    [U0,~,V0] = svd(W*P','econ');
    Q = U0*V0';
    QP = Q*P;
    clear U0 V0;
    
    % Optimize matrix W.
    XXT = X*X' + (beta+gamma)*eye(dim);
    R = XXT+lambda*fea*L*fea';
    W = R\(X*T'+gamma*QP);
    b = T - W' * X;
    b = mean(b,2);
    
    % Optimize matrix T
    U = W'*X + b*ones(1,N);
    for ind = 1:N
        T(:,ind) = optimizedT(U(:,ind), class_id(ind));
    end
    clear R U;

    % Optimize matrix A
    wfea = W'*fea;
    distx = L2_distance_1(wfea,wfea);
    if iter>1
        [~, idx] = sort(distx,2);
    end;
    A = zeros(tolN);
    for j=1:tolN
        idxa0 = idx(j,2:knum+1);
        dxi = distx(j,idxa0);
        ad = -dxi/(2*r);
        A(j,idxa0) = EProjSimplex_new(ad);
    end;
    A = (A+A')/2;
    D = diag(sum(A));
    L = D-A;
    clear distx A D idxa0 dxi ad
      
    T1 = W'*X + b*ones(1,N)-T;
    T2 = trace(wfea*L*wfea');
    obj(iter)=trace(T1'*T1)+lambda*T2; clear T1 T2;
    disp(['iter = ' num2str(iter),  ' Obj = ' num2str(obj(iter)) ]);
end
end

function T = optimizedT(R, label)
classNum = length(R);
T = zeros(classNum,1);
V = R + 1 - repmat(R(label),classNum,1);    
step = 0;
num = 0;
for i = 1:classNum
    if i~=label
        dg = V(i);
        for j = 1:classNum;
            if j~=label
                if V(i) < V(j)
                    dg = dg + V(i) - V(j);
                end
            end
        end
        if dg > 0
            step = step + V(i);
            num = num + 1;
        end
    end
end
step = step / (1+num);
for i = 1:classNum
    if i == label
        T(i) = R(i) + step;
    else
        T(i) = R(i) + min(step - V(i), 0);
    end
end
end
