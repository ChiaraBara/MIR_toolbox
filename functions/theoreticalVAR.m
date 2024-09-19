%% THEORETICAL COEFFICIENTS FOR SIMULATED VAR PROCESSES
% Give theoretical model coefficient from number of simulated processes M
% and a structure par with the parameter of the processes, i.e.,
%   par.poles=([amplitude frequency)] of the oscillations;
%   par.coup=([i j k c;...]) impose a coupling from i to j at lag k with
%   strength c;
%   par.Su=[] variance of the predizion error.
% The coefficient matrix Ak (and its concatened form Am) and the variance
% of the predicition error are obtained.
function [Am,Su,Ak]=theoreticalVAR(M,par)

Su=eye(M); %innovation covariance matrix (diagonal)
for k=1:M
    Su(k,k)=par.Su(k);
end

for m=1:M
    npoles=size(par.poles{m},1);
    z{m}=[]; % list of poles
    for n=1:npoles
        r=par.poles{m}(n,1);
        f=par.poles{m}(n,2);
        if f==0 % add real pole
            z{m}=[z{m}; r];
        else %add complex conjugate pole
            za=r*(cos(2*pi*f)+1i*sin(2*pi*f)); %pole
            zb=r*(cos(2*pi*f)-1i*sin(2*pi*f)); %complex conjugate pole
            z{m}=[z{m}; za; zb];
        end
    end
    Apol=poly(z{m});
    Amd{m}=-Apol(2:length(Apol));
    pd(m)=length(Amd{m}); %max delay for autonomous oscillations in process m
end

if isempty(par.coup)
    p=max(pd);
else
    p=max(max(pd),max(par.coup(:,3))); %maximum delay
end

Ak=zeros(M,M,p); %blocks of coefficients
for m=1:M
    for k=1:length(Amd{m})
        Ak(m,m,k)=Amd{m}(k);
    end
end
for k=1:size(par.coup,1)
    Ak(par.coup(k,1),par.coup(k,2),par.coup(k,3))=par.coup(k,4);
end

Am=[]; %group coefficient blocks Ak in a single matrix Am
for kk=1:p
    Am=[Am; Ak(:,:,kk)];
end

% stability check
Am1=Am';
E=eye(M*p);AA=[Am1;E(1:end-M,:)];lambda=eig(AA);lambdamax=max(abs(lambda));
if lambdamax>=1
    error('The simulated VAR process is not stable');
end


end