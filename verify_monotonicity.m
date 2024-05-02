function [viol] = verify_monotonicity(data,measure)

[Nalt, Ncrit] = size(data);

for ii=1:Nalt
    for jj=1:Nalt
        if ii~=jj
            teste1(ii,jj) = (sum(data(ii,:)>=data(jj,:))==Ncrit);
        end
    end
end

for ii=1:Nalt
    for jj=1:Nalt
        if ii~=jj
            teste2(ii,jj) = measure(ii)>=measure(jj);
        end
    end
end

matrix = teste1.*teste2 == teste1;
[rviol,cviol] = find(matrix==0);
viol = [rviol,cviol];

end

