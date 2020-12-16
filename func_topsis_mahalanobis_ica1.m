%% Function that calculates the TOPSIS-Mahalanobis ordering with:
% PIS and NIS provided by a BSS approach
% Vector normalization
% Weighted evaluations

function [sim,order,norm_rat,pis,nis] = func_topsis_mahalanobis_ica1(Nalt,data,weights,pis,nis)

exc = size(data,1) - Nalt; % Indicate if PIS and NIS is in the data

% Calculating normalized ratings
norm_rat = data./repmat(sqrt(sum(data(1:Nalt,:).^2)),Nalt+exc,1);

% Defining diagonal matrix of weights
Dweights = diag(weights);

% Determining positive and negative ideal points
pis = pis./sqrt(sum(data(1:Nalt,:).^2));
nis = nis./sqrt(sum(data(1:Nalt,:).^2));

% Calculating the inverse of the covariance matrix
C_inv = inv(cov(norm_rat(1:Nalt,:)));

% Calculating the distance from PIS and NIS
dist_pis = zeros(Nalt,1);
for ii=1:Nalt
    dist_pis(ii) = sqrt((norm_rat(ii,:)-pis)*Dweights*C_inv*Dweights*(norm_rat(ii,:)-pis)');
end
dist_nis = zeros(Nalt,1);
for ii=1:Nalt
    dist_nis(ii) = sqrt((norm_rat(ii,:)-nis)*Dweights*C_inv*Dweights*(norm_rat(ii,:)-nis)');
end

% Calculating similarities
sim = dist_nis./(dist_pis + dist_nis);

% Determining the ordering
[~,order] = sort(sim,'descend');

end

