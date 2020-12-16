function [W, Y] = ICA_infomax_fastICA(N,M,Z);

% Esta função encontra os ICs usando a INFOMAX junto ao método do ponto 
% fixo.

% %Branqueamento; Z: [N+1,M] matriz branca
% m1 =(mean(X(1,:))); m1 = repmat(m1,1,M);
% m2 =(mean(X(2,:))); m2 = repmat(m2,1,M);
% X(1,:) = X(1,:) - m1;
% X(2,:) = X(2,:) - m2;
% covar = cov(X');
% [eigve,eigva] = eig(covar);
% V = eigve*eigva^(-0.5)*eigve';
% Z = V'*X;
% correl = corr(Z');
% Z(:,1) = Z(:,1)/((dot(Z(:,1),Z(:,1)))^(0.5));
% Z(:,2) = Z(:,2)/((dot(Z(:,2),Z(:,2)))^(0.5));

%Z = X;

W = rand(N+1,N+1);

iter = 2000;
sk = 10^(-1);
ws = zeros(iter*2,N+1);

for ii = 1:iter
    
    y = W*Z;
    
    %g = -(W*Z).^3;
    
     g = tanh(y) - y;
    
    W = W + sk*(inv(W') + (g*Z')/M);
    
    ws((N+1)*ii-N:(N+1)*ii,:) = W;
    
    for jj=1:N
    
    W(:,jj) = W(:,jj)/((dot(W(:,jj),W(:,jj)))^(0.5));
    
    end
    %W = W + sk*(inv(W') + ((W*Z).*(W*Z).*(W*Z)*Z')/M);
    
end

Y = W*Z;