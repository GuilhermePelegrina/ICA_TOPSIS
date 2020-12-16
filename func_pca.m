function [Z,V] = func_pca(Nalt,data)

% Zeroing the mean of the data
data = data';
m = mean(data');
data = data - repmat(m',1,Nalt);

% Whitening
covar = cov(data');
[eigve,eigva] = eig(covar);
V = eigve*eigva^(-0.5)*eigve';
Z = V'*data;
Z = Z';

% Z1(:,1) = Z(:,1)/((dot(Z(:,1),Z(:,1)))^(0.5));
% Z1(:,2) = Z(:,2)/((dot(Z(:,2),Z(:,2)))^(0.5));
% Z1(:,3) = Z(:,3)/((dot(Z(:,3),Z(:,3)))^(0.5));
% Z1(:,4) = Z(:,4)/((dot(Z(:,4),Z(:,4)))^(0.5));
% Z1(:,5) = Z(:,5)/((dot(Z(:,5),Z(:,5)))^(0.5));
% Z1(:,6) = Z(:,6)/((dot(Z(:,6),Z(:,6)))^(0.5));

end
