%% quantizes the series y with c quantization levels
% y: input series, column data
% c: number of quantization levels

function x=MIR_quantization(y,c)

n=size(y,1);

x=zeros(n,1);
ma=max(y); mi=min(y);
q=(ma-mi)/c; % amplitude of quantization level

l=zeros(c,1);
for i=1:c %quantization levels
   l(i)=mi+i*q;
end

for i=1:n
   j=1;
   while (y(i)>=l(j))&&(j<c)
      j=j+1;
   end
   x(i)=j;
end

end

