%% TOPSIS with Mahalanobis distance - Analysis with respect to ICA
clear all; close all; clc;

Niter = 500;
alpha = [-0.75:0.05:0.75];
beta = [-0.75:0.05:0.75];
Ncrit = 2;
Nalt = 100;
weights = ones(1,Ncrit)*(1/Ncrit);

kendall = zeros(Niter,length(alpha),length(beta));

%% Data (generated at random) and Parameters
% It is needed to change the size of the dataset and the convergence
% criteria of FastICA

for tt=1:Niter
    
    data = 1*rand(Nalt,Ncrit);
    [data,~] = func_pca(Nalt,data);
    
    % TOPSIS (euclidian) in original data
    [simEO,orderEO,dataEO,pisEO,nisEO] = func_topsis_euclidean1(Nalt,data,weights); % Euclidian - original data
    orderEO_pos = orderEO; orderEO_pos(orderEO) = 1:Nalt; % Original (correct) ordering
    
    for kk=1:length(alpha)
        for ll=1:length(beta)
            
            Mmixing = [1 alpha(kk); beta(ll) 1];
            data_mix = (Mmixing*data')'; % Generate the mixed data
            
            % Additive noise
%                         for ii=1:Ncrit
%                             data_mix(:,ii) = awgn(data_mix(:,ii),35);
%                         end
            
            [simMM,orderMM,dataMM,pisMM,nisMM] = func_topsis_mahalanobis1(Nalt,data_mix,weights); % Mahalanobis - mixed data
            
            %% Ordering
            orderMM_pos = orderMM; orderMM_pos(orderMM) = 1:Nalt; % Mixed (mahalanobis) ordering
            
            %% Kendall Tau distance (full rank)
%             kendall(tt,kk,ll) = normalized_kendalltau(orderMM,orderEO);
            kendall(tt,kk,ll) = corr(orderMM_pos,orderEO_pos,'type','Kendall');
%             kendall(tt,kk,ll) = corr(simMM,simEO);
        
        end
    end
    tt
%     save('resultado_mahalanobis','kendall', 'alpha', 'beta', 'tt');
end

kendall_mean = mean(kendall); kendall_mean = squeeze(kendall_mean);
[Xalpha,Yalpha] = meshgrid(alpha,beta);
% figure; surface(Xalpha,Yalpha,kendall_mean);
figure; surfc(Xalpha,Yalpha,kendall_mean); xlabel('\alpha'); ylabel('\beta')
colormap(flipud(gray)); caxis([0.5 1]); colorbar;

% figure; mesh(Xalpha,Yalpha,kendall_mean); colormap jet;
% hold on; contour(Xalpha,Yalpha,kendall_mean);
