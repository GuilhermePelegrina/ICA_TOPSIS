%% Function that calculates the TOPSIS ordering with:
% Vector normalization
% Weighted evaluations

function [sim,order,w_norm_ra,pis,nis] = func_topsis_euclidean1(Nalt,data,weights)

exc = size(data,1) - Nalt; % Indicate if PIS and NIS is in the data

% Calculating normalized ratings
norm_rat = data./repmat(sqrt(sum(data(1:Nalt,:).^2)),Nalt+exc,1);

% Calculating wheighted normalized ratings
w_norm_ra = repmat(weights,Nalt+exc,1).*norm_rat;

% Determining positive and negative ideal points
pis = max(w_norm_ra(1:Nalt,:));
nis = min(w_norm_ra(1:Nalt,:));

% Calculating the distance from PIS and NIS
dist_pis = (sum((w_norm_ra(1:Nalt,:) - repmat(pis,Nalt,1)).^2,2)).^(1/2);
dist_nis = (sum((w_norm_ra(1:Nalt,:) - repmat(nis,Nalt,1)).^2,2)).^(1/2);

% Calculating similarities
sim = dist_nis./(dist_pis + dist_nis);

% Determining the ordering
[~,order] = sort(sim,'descend');

end

