%% k-nearest neighbor Estimation of the Bivariate Transfer Entropy
% Y: bivariate time series (N x 2)
% m: memory (number of past samples in the embedding vectors)
% tau: vector of embedding delays (one for each series)
% k: number of neighbors
% metric: 'maximum' Chebyshev distance (default)

function ret=MIR_MIRdec_knn(Y,m,tau,k,metric)

M=size(Y,2);
V=MIR_SetLag([m m],tau,ones(1,M),[0 0]);

if ~exist('metric','var'), metric='maximum'; end

%% form the observation matrices
B_jj=MIR_ObsMat(Y,2,V);
B_ii=MIR_ObsMat(Y,1,V);
A_jj=B_jj(:,2:end); 
A_ii=B_ii(:,2:end);
tmp=V(:,1);

i_Y= tmp==2;
i_X= tmp==1;

M_y=B_jj(:,1);
M_Y=A_jj(:,i_Y);
M_x=B_ii(:,1); 
M_X=A_ii(:,i_X);

M_yY=[M_y M_Y];
M_xX=[M_x M_X];
M_YX=[M_Y M_X];
M_yYX=[M_y M_YX];
M_xYX=[M_x M_YX];
M_yxYX=[M_y M_xYX];

N=size(B_jj,1);

%% kNN analysis
%%% neighbor search in space of higher dimension
atria_yxYX = nn_prepare(M_yxYX, metric);
[~, distances] = nn_search(M_yxYX, atria_yxYX, (1:N)', k, 0);
dd=distances(:,k);

%%% range searches in subspaces of lower dimension - M_X
if ~isempty(M_X)
    atria_X = nn_prepare(M_X, metric);
    [count_X, tmp] = range_search(M_X, atria_X, (1:N)', dd, 0);
    tmp=tmp(:,2);%subtraction from count of points with distance exactly equal to k-th neighbor
    for n=1:length(tmp)
        count_X(n)=max(k-1,count_X(n)-sum(tmp{n}==dd(n)));
    end
else
    count_X=(N-1)*ones(N,1);
end

%%% range searches in subspaces of lower dimension - M_xX
if ~isempty(M_xX)
    atria_xX = nn_prepare(M_xX, metric);
    [count_xX, tmp] = range_search(M_xX, atria_xX, (1:N)', dd, 0);
    tmp=tmp(:,2);%subtraction from count of points with distance exactly equal to k-th neighbor
    for n=1:length(tmp)
        count_xX(n)=max(k-1,count_xX(n)-sum(tmp{n}==dd(n)));
    end
else
    count_xX=(N-1)*ones(N,1);
end

%%% range searches in subspaces of lower dimension - M_Y
if ~isempty(M_Y)
    atria_Y = nn_prepare(M_Y, metric);
    [count_Y, tmp] = range_search(M_Y, atria_Y, (1:N)', dd, 0);
    tmp=tmp(:,2);%subtraction from count of points with distance exactly equal to k-th neighbor
    for n=1:length(tmp)
        count_Y(n)=max(k-1,count_Y(n)-sum(tmp{n}==dd(n)));
    end
else
    count_Y=(N-1)*ones(N,1);
end

%%% range searches in subspaces of lower dimension - M_yY
if ~isempty(M_yY)
    atria_yY = nn_prepare(M_yY, metric);
    [count_yY, tmp] = range_search(M_yY, atria_yY, (1:N)', dd, 0);
    tmp=tmp(:,2); %subtraction from count of points with distance exactly equal to k-th neighbor
    for n=1:length(tmp)
        count_yY(n)=max(k-1,count_yY(n)-sum(tmp{n}==dd(n)));
    end
else
    count_yY=(N-1)*ones(N,1);
end

%%% range searches in subspaces of lower dimension - M_YX
if ~isempty(M_YX)
    atria_YX = nn_prepare(M_YX, metric);
    [count_YX, tmp] = range_search(M_YX, atria_YX, (1:N)', dd, 0);
    tmp=tmp(:,2);%subtraction from count of points with distance exactly equal to k-th neighbor
    for n=1:length(tmp)
        count_YX(n)=max(k-1,count_YX(n)-sum(tmp{n}==dd(n)));
    end
else
    count_YX=(N-1)*ones(N,1);
end

%%% range searches in subspaces of lower dimension - M_yYX
if ~isempty(M_yYX)
    atria_yYX = nn_prepare(M_yYX, metric);
    [count_yYX, tmp] = range_search(M_yYX, atria_yYX, (1:N)', dd, 0);
    tmp=tmp(:,2);%subtraction from count of points with distance exactly equal to k-th neighbor
    for n=1:length(tmp)
        count_yYX(n)=max(k-1,count_yYX(n)-sum(tmp{n}==dd(n)));
    end
else
    count_yYX=(N-1)*ones(N,1);
end

%%% range searches in subspaces of lower dimension - M_xYX
if ~isempty(M_xYX)
    atria_xYX = nn_prepare(M_xYX, metric);
    [count_xYX, tmp] = range_search(M_xYX, atria_xYX, (1:N)', dd, 0);
    tmp=tmp(:,2);%subtraction from count of points with distance exactly equal to k-th neighbor
    for n=1:length(tmp)
        count_xYX(n)=max(k-1,count_xYX(n)-sum(tmp{n}==dd(n)));
    end
else
    count_xYX=(N-1)*ones(N,1);
end

%% compute MIR
dd2=2*dd; dd2(dd2==0)=[];

Hx_X = mean(log(dd2)) + (1/N)*(sum(psi(count_X+1)-psi(count_xX+1)));
Hy_Y = mean(log(dd2)) + (1/N)*(sum(psi(count_Y+1)-psi(count_yY+1)));
Hxy_XY = -psi(k)+2*mean(log(dd2))+(1/N)*(sum(psi(count_YX+1)));
I_XY = Hx_X + Hy_Y - Hxy_XY;

T_XY = (1/N)*( sum(psi(count_Y+1)) - sum(psi(count_yY+1)) - sum(psi(count_YX+1)) + sum(psi(count_yYX+1)));
T_YX = (1/N)*( sum(psi(count_X+1)) - sum(psi(count_xX+1)) - sum(psi(count_YX+1)) + sum(psi(count_xYX+1)));
I_XoY = psi(k) + (1/N)*( sum(psi(count_YX+1)) - sum(psi(count_yYX+1)) - sum(psi(count_xYX+1)) );
I_XY2 = T_XY + T_YX + I_XoY;

%% output
ret.I_XY=I_XY; 
ret.Hx_X=Hx_X; 
ret.Hy_Y=Hy_Y; 
ret.Hxy_XY = Hxy_XY;

ret.I_XY2=I_XY2; 
ret.T_XY=T_XY; 
ret.T_YX=T_YX; 
ret.I_XoY=I_XoY; 

end
