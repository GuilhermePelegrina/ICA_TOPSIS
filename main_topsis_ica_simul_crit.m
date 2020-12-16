%% ICA TOPSIS analysis varying the number of criteria
% Comparison considering the following approaches:
%   TOPSIS-E in Mixed data
%   TOPSIS-M in mixed data
%   ICA-TOPSIS-M (JADE)
% Apply vector normalization in decision data
% Mixing matrix generated with diag=1 and off-diagonal=rand in [-0.75,-0.25] or [0.25, 0.75]
% SNR equals to 15, 30 or 45
% Number of alternatives equals to 30, 100 or 170
% Number of criteria equals to 3, 4 or 5
% Parameters to change:
%   Weighted evaluations (...euclidean/mahalanobis1) or Weighted distances (...euclidean/mahalanobis2)
%   Number of criteria (Ncrit). Attention for the FastICA convergence condition
%   Vector of weights
%   Number of iterations
clear all; close all; clc;

snr = [15 30 45]; % Range of Signal-to-Noise ratio applyied in mixed data
Nalt = [30 100 170]; % # of alternatives
Ncrit = 5; % # of criteria
weights = ones(1,Ncrit)*(1/Ncrit); % Equal weights
Niter = 1000; % # of iterations

coefcorr1 = zeros(Niter,length(snr),length(Nalt)); % Matrix which will store the Pearson correlation coeficient
kendallcorr1 = zeros(Niter,length(snr),length(Nalt)); % Matrix which will store the Kendall tau coeficient
error_first1 = zeros(Niter,length(snr),length(Nalt)); % Matrix which will store the total absolute error in the first 20% positions
error_last1 = zeros(Niter,length(snr),length(Nalt)); % Matrix which will store the total absolute error in the last 20% positions

coefcorr2 = zeros(Niter,length(snr),length(Nalt)); % Matrix which will store the Pearson correlation coeficient
kendallcorr2 = zeros(Niter,length(snr),length(Nalt)); % Matrix which will store the Kendall tau coeficient
error_first2 = zeros(Niter,length(snr),length(Nalt)); % Matrix which will store the total absolute error in the first 20% positions
error_last2 = zeros(Niter,length(snr),length(Nalt)); % Matrix which will store the total absolute error in the last 20% positions

coefcorr3 = zeros(Niter,length(snr),length(Nalt)); % Matrix which will store the Pearson correlation coeficient
kendallcorr3 = zeros(Niter,length(snr),length(Nalt)); % Matrix which will store the Kendall tau coeficient
error_first3 = zeros(Niter,length(snr),length(Nalt)); % Matrix which will store the total absolute error in the first 20% positions
error_last3 = zeros(Niter,length(snr),length(Nalt)); % Matrix which will store the total absolute error in the last 20% positions

for tt=1:Niter
    kk = 1; % Counter
    while kk <= length(snr)
        pp = 1;
        while pp <= length(Nalt)
            %% Data and mixing matrix (generated at random)
            data = 1*rand(Nalt(pp),Ncrit);
            Mmixing = sign(rand(Ncrit,Ncrit)-0.5).*(0.3*rand(Ncrit,Ncrit)+0.2); Mmixing(1:Ncrit+1:end) = 1;
            
            %% TOPSIS (Euclidean) in original data
            [simEO,orderEO,~,~,~] = func_topsis_euclidean1(Nalt(pp),data,weights);
            
            %% Mixed data
            data_mix = (Mmixing*data')'; % Generate the mixed data
            for ii=1:Ncrit % Additive White Gaussian Noise
                data_mix(:,ii) = awgn(data_mix(:,ii),snr(kk),'measured');
            end
            
            %% TOPSIS (Euclidean) in mixed data
            [simEM,orderEM,~,~,~] = func_topsis_euclidean1(Nalt(pp),data_mix,weights);
            
            %% Extracting independent data (ICA-JADE)
            
            % Jade
            B = jadeR(data_mix',Ncrit); data_jade = B*data_mix'; % Jade algorithm
            data_jade = data_jade';
            [data_jade_adj,matrix_mixing_jade] = func_bss_correct(Nalt(pp),Ncrit,inv(B),data_jade); % Correcting ambiguities
            
            %% TOPSIS (Mahalanobis) in mixed data
            Dweights = diag(weights); % Defining diagonal matrix of weights
            norm_rat = data_mix./repmat(sqrt(sum(data_mix(1:Nalt(pp),:).^2)),Nalt(pp),1); % Calculating normalized ratings
            C_inv = inv(cov(norm_rat(1:Nalt(pp),:))); % Calculating the inverse of the covariance matrix
            L = chol(inv(C_inv), 'lower'); % Cholesky factorization
            
            % Original TOPSIS-M
            [simMM,orderMM,~,~,~] = func_topsis_mahalanobis1(Nalt(pp),data_mix,weights); % Mixed data
            
            % ICA - Jade
            pis_jade = max(data_jade_adj); pis_jade_transf = (matrix_mixing_jade*pis_jade')';
            nis_jade = min(data_jade_adj); nis_jade_transf = (matrix_mixing_jade*nis_jade')';
            [simMM_ica_jade,orderMM_ica_jade,~,~,~] = func_topsis_mahalanobis_ica1(Nalt(pp),data_mix,weights,pis_jade_transf,nis_jade_transf); % Jade
            
            %% Ordering and Kendall tau and Pearson coeficients
            orderEO_pos = orderEO; orderEO_pos(orderEO) = 1:Nalt(pp); % Original (correct) ordering
            
            orderEM_pos = orderEM; orderEM_pos(orderEM) = 1:Nalt(pp); % Mixed (Euclidean) ordering
            coefcorr1(tt,kk,pp) = corr(simEM,simEO);
            kendallcorr1(tt,kk,pp) = corr(orderEM_pos,orderEO_pos,'type','Kendall');
            
            orderMM_pos = orderMM; orderMM_pos(orderMM) = 1:Nalt(pp); % Mixed (Mahalanobis) ordering
            coefcorr2(tt,kk,pp) = corr(simMM,simEO);
            kendallcorr2(tt,kk,pp) = corr(orderMM_pos,orderEO_pos,'type','Kendall');
            
            orderMM_ica_jade_pos = orderMM_ica_jade; orderMM_ica_jade_pos(orderMM_ica_jade) = 1:Nalt(pp); % Mixed (Mahalanobis+jade) ordering
            coefcorr3(tt,kk,pp) = corr(simMM_ica_jade,simEO);
            kendallcorr3(tt,kk,pp) = corr(orderMM_ica_jade_pos,orderEO_pos,'type','Kendall');
            
            % Error first/last 20% of positions
            for ll=1:round(0.2*Nalt(pp))
                error_first1(tt,kk,pp) = error_first1(tt,kk,pp) + abs(ll-orderEM_pos(orderEO(ll)));
                error_first2(tt,kk,pp) = error_first2(tt,kk,pp) + abs(ll-orderMM_pos(orderEO(ll)));
                error_first3(tt,kk,pp) = error_first3(tt,kk,pp) + abs(ll-orderMM_ica_jade_pos(orderEO(ll)));
                
                error_last1(tt,kk,pp) = error_last1(tt,kk,pp) + abs(Nalt(pp)+1-ll-orderEM_pos(orderEO(Nalt(pp)+1-ll)));
                error_last2(tt,kk,pp) = error_last2(tt,kk,pp) + abs(Nalt(pp)+1-ll-orderMM_pos(orderEO(Nalt(pp)+1-ll)));
                error_last3(tt,kk,pp) = error_last3(tt,kk,pp) + abs(Nalt(pp)+1-ll-orderMM_ica_jade_pos(orderEO(Nalt(pp)+1-ll)));
            end
            
            pp = pp + 1; % Update the counter
        end
        kk = kk + 1; % Update the counter
    end
    tt
    %save('resultado_simul_crit_5crit','coefcorr1','coefcorr2','coefcorr3', 'kendallcorr1','kendallcorr2','kendallcorr3', 'snr', 'Nalt', 'error_first1','error_first2','error_first3', 'error_last1', 'error_last2','error_last3', 'tt');
end

error_first1(:,:,1) = error_first1(:,:,1)/Nalt(1); error_first1(:,:,2) = error_first1(:,:,2)/Nalt(2); error_first1(:,:,3) = error_first1(:,:,3)/Nalt(3);
error_first2(:,:,1) = error_first2(:,:,1)/Nalt(1); error_first2(:,:,2) = error_first2(:,:,2)/Nalt(2); error_first2(:,:,3) = error_first2(:,:,3)/Nalt(3);
error_first3(:,:,1) = error_first3(:,:,1)/Nalt(1); error_first3(:,:,2) = error_first3(:,:,2)/Nalt(2); error_first3(:,:,3) = error_first3(:,:,3)/Nalt(3);

error_last1(:,:,1) = error_last1(:,:,1)/Nalt(1); error_last1(:,:,2) = error_last1(:,:,2)/Nalt(2); error_last1(:,:,3) = error_last1(:,:,3)/Nalt(3);
error_last2(:,:,1) = error_last2(:,:,1)/Nalt(1); error_last2(:,:,2) = error_last2(:,:,2)/Nalt(2); error_last2(:,:,3) = error_last2(:,:,3)/Nalt(3);
error_last3(:,:,1) = error_last3(:,:,1)/Nalt(1); error_last3(:,:,2) = error_last3(:,:,2)/Nalt(2); error_last3(:,:,3) = error_last3(:,:,3)/Nalt(3);

cc1 = mean(coefcorr1); cc2 = mean(coefcorr2); cc3 = mean(coefcorr3);
ck1 = mean(kendallcorr1); ck2 = mean(kendallcorr2); ck3 = mean(kendallcorr3);
ef1 = mean(error_first1); ef2 = mean(error_first2); ef3 = mean(error_first3);
el1 = mean(error_last1); el2 = mean(error_last2); el3 = mean(error_last3);

cc1s = std(coefcorr1); cc2s = std(coefcorr2); cc3s = std(coefcorr3);
ck1s = std(kendallcorr1); ck2s = std(kendallcorr2); ck3s = std(kendallcorr3);
ef1s = std(error_first1); ef2s = std(error_first2); ef3s = std(error_first3);
el1s = std(error_last1); el2s = std(error_last2); el3s = std(error_last3);
