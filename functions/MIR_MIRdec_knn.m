%% k-nearest neighbor Estimation of the Bivariate Transfer Entropy
% Y: bivariate time series (N x 2)
% m: memory (number of past samples in the embedding vectors)
% tau: vector of embedding delays (one for each series)
% k: number of neighbors
% metric: 'maximum' Chebyshev distance (default)

function ret=MIR_MIRdec_knn(Y,m,tau,k,metric)

M=size(Y,2);
V=MIR_SetLag([m m],tau,ones(1,M),[0 0]);

if ~exist('metric','var'), metric='chebychev'; end

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
[~, distances] =  knnsearch(M_yxYX,M_yxYX,'K',k+1,'Distance',metric);
dd = distances(:,end);

%%% range searches in subspaces of lower dimension - M_X
[~, distance_X] =  knnsearch(M_X,M_X,'K',N,'Distance',metric);
count_X = distance_X(:,2:end) < dd;
count_X = max(k-1, sum(count_X,2));

%%% range searches in subspaces of lower dimension - M_xX
[~, distance_xX] =  knnsearch(M_xX,M_xX,'K',N,'Distance',metric);
count_xX = distance_xX(:,2:end) < dd;
count_xX = max(k-1, sum(count_xX,2));

%%% range searches in subspaces of lower dimension - M_Y
[~, distance_Y] =  knnsearch(M_Y,M_Y,'K',N,'Distance',metric);
count_Y = distance_Y(:,2:end) < dd;
count_Y = max(k-1, sum(count_Y,2));

%%% range searches in subspaces of lower dimension - M_yY
[~, distance_yY] =  knnsearch(M_yY,M_yY,'K',N,'Distance',metric);
count_yY = distance_yY(:,2:end) < dd;
count_yY = max(k-1, sum(count_yY,2));

%%% range searches in subspaces of lower dimension - M_YX
[~, distance_YX] =  knnsearch(M_YX,M_YX,'K',N,'Distance',metric);
count_YX = distance_YX(:,2:end) < dd;
count_YX = max(k-1, sum(count_YX,2));

%%% range searches in subspaces of lower dimension - M_yYX
[~, distance_yYX] =  knnsearch(M_yYX,M_yYX,'K',N,'Distance',metric);
count_yYX = distance_yYX(:,2:end) < dd;
count_yYX = max(k-1, sum(count_yYX,2));

%%% range searches in subspaces of lower dimension - M_xYX
[~, distance_xYX] =  knnsearch(M_xYX,M_xYX,'K',N,'Distance',metric);
count_xYX = distance_xYX(:,2:end) < dd;
count_xYX = max(k-1, sum(count_xYX,2));

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
