%% ICA TOPSIS analysis varying the SNR
% Comparison considering the following approaches:
%   TOPSIS-E in Mixed data
%   TOPSIS-M in mixed data
%   ICA-TOPSIS (FastICA, Infomax and Jade)
%   ICA-TOPSIS-M
% Apply vector normalization in decision data
% Mixing matrix generated with diag=1 and off-diagonal=rand in [-0.75,-0.25] or [0.25, 0.75]
% SNR in the range (0,50]
% Parameters to change:
%   Weighted evaluations (...euclidean/mahalanobis1) or Weighted distances (...euclidean/mahalanobis2)
%   Number of alternatives (Nalt)
%   Number of criteria (Ncrit). Attention for the FastICA convergence condition
%   Vector of weights
%   Number of iterations
clear all; close all; clc;

Nalt = 100; % # of alternatives
Ncrit = 2; % # of criteria
weights = ones(1,Ncrit)*(1/Ncrit); % Equal weights
snr = [0:1:50]; % Range of Signal-to-Noise ratio applyied in mixed data
Niter = 500; % # of iterations

sir = zeros(Niter,length(snr),3); % Matrix which will store the Amari metrics
metrics = zeros(Niter,length(snr),3); % Matrix which will store the Amari metrics
coefcorr = zeros(Niter,length(snr),10); % Matrix which will store the Pearson correlation coeficient
kendallcorr = zeros(Niter,length(snr),10); % Matrix which will store the Kendall tau coeficient
error_first = zeros(Niter,length(snr),10); % Matrix which will store the total absolute error in the first 20% positions
error_last = zeros(Niter,length(snr),10); % Matrix which will store the total absolute error in the last 20% positions

for tt=1:Niter
    kk = 1; % Counter
    while kk <= length(snr)
        %% Data and mixing matrix (generated at random)
        data = 1*rand(Nalt,Ncrit);
        Mmixing = sign(rand(Ncrit,Ncrit)-0.5).*(0.3*rand(Ncrit,Ncrit)+0.2); Mmixing(1:Ncrit+1:end) = 1;
        
        %% TOPSIS (Euclidean) in original data
        [simEO,orderEO,~,~,~] = func_topsis_euclidean1(Nalt,data,weights);
        
        %% Mixed data
        data_mix = (Mmixing*data')'; % Generate the mixed data
        for ii=1:Ncrit % Additive White Gaussian Noise
            data_mix(:,ii) = awgn(data_mix(:,ii),snr(kk),'measured');
        end
        
        %% TOPSIS (Euclidean) in mixed data
        [simEM,orderEM,~,~,~] = func_topsis_euclidean1(Nalt,data_mix,weights);
        
        %% Extracting independent data (ICA approaches)
        
        % FastICA
        [data_fastica,matrix_mixing,matrix_separating] = fastica(data_mix(1:Nalt,:)','numOfIC',Ncrit); % FastICA algorithm
        data_fastica = data_fastica';
        [data_fastica_adj,matrix_mixing_fastica] = func_bss_correct(Nalt,Ncrit,matrix_mixing,data_fastica); % Correcting ambiguities
                if isempty(data_fastica); %
%         if isempty(data_fastica) || size(data_fastica,2) < 3;
            %         if isempty(data_fastica) || size(data_fastica,2) < 4;
        else
            [metrics_fastica] = amari_metrics(matrix_separating,Ncrit,Mmixing); % Amari metric
            
            % Infomax
            [Z,V] = func_pca(Nalt,data_mix); % Whitening
            [W, data_infomax] = ICA_infomax_fastICA(Ncrit-1,Nalt,Z'); % Infomax algorithm
            data_infomax = data_infomax';
            data_infomax = (W*V'*data_mix')'; % Correcting the mean
            [data_infomax_adj,matrix_mixing_infomax] = func_bss_correct(Nalt,Ncrit,inv(W*V),data_infomax); % Correcting ambiguities
            [metrics_infomax] = amari_metrics(inv(matrix_mixing_infomax),Ncrit,Mmixing); % Amari metric
            
            % Jade
            B = jadeR(data_mix',Ncrit); data_jade = B*data_mix'; % Jade algorithm
            data_jade = data_jade';
            [data_jade_adj,matrix_mixing_jade] = func_bss_correct(Nalt,Ncrit,inv(B),data_jade); % Correcting ambiguities
            [metrics_jade] = amari_metrics(B,Ncrit,Mmixing); % Amari metric
            
            % Utopic separation (knowing the mixing matrix)
            data_utopic = inv(Mmixing)*data_mix';
            data_utopic = data_utopic';
            
            %% TOPSIS (Euclidean) in independent data
            [simEI_f,orderEI_f,~,~,~] = func_topsis_euclidean1(Nalt,data_fastica_adj,weights); % FastICA
            [simEI_i,orderEI_i,~,~,~] = func_topsis_euclidean1(Nalt,data_infomax_adj,weights); % Infomax
            [simEI_j,orderEI_j,~,~,~] = func_topsis_euclidean1(Nalt,data_jade_adj,weights); % Jade
            [simEI_utopic,order_utopic,~,~,~] = func_topsis_euclidean1(Nalt,data_utopic,weights); % Jade
            
            %% TOPSIS (Mahalanobis) in mixed data
            Dweights = diag(weights); % Defining diagonal matrix of weights
            norm_rat = data_mix./repmat(sqrt(sum(data_mix(1:Nalt,:).^2)),Nalt,1); % Calculating normalized ratings
            C_inv = inv(cov(norm_rat(1:Nalt,:))); % Calculating the inverse of the covariance matrix
            L = chol(inv(C_inv), 'lower'); % Cholesky factorization
            
            % Original TOPSIS-M
            [simMM,orderMM,~,~,~] = func_topsis_mahalanobis1(Nalt,data_mix,weights); % Mixed data
            
            % Utopic TOPSIS-M (with correct PIS and NIS transformed into mixed data)
            pis_utopic = max(data); pis_utopic_transf = (Mmixing*pis_utopic')';
            nis_utopic = min(data); nis_utopic_transf = (Mmixing*nis_utopic')';
            [simMM_utopic,orderMM_utopic,~,~,~] = func_topsis_mahalanobis_ica1(Nalt,data_mix,weights,pis_utopic_transf,nis_utopic_transf); % Utopic
            
            % ICA - FastICA
            pis_fastica = max(data_fastica_adj); pis_fastica_transf = (matrix_mixing_fastica*pis_fastica')';
            nis_fastica = min(data_fastica_adj); nis_fastica_transf = (matrix_mixing_fastica*nis_fastica')';
            [simMM_ica_fastica,orderMM_ica_fastica,~,~,~] = func_topsis_mahalanobis_ica1(Nalt,data_mix,weights,pis_fastica_transf,nis_fastica_transf); % FastICA
            
            % ICA - Infomax
            pis_infomax = max(data_infomax_adj); pis_infomax_transf = (matrix_mixing_infomax*pis_infomax')';
            nis_infomax = min(data_infomax_adj); nis_infomax_transf = (matrix_mixing_infomax*nis_infomax')';
            [simMM_ica_infomax,orderMM_ica_infomax,~,~,~] = func_topsis_mahalanobis_ica1(Nalt,data_mix,weights,pis_infomax_transf,nis_infomax_transf); % Infomax
            
            % ICA - Jade
            pis_jade = max(data_jade_adj); pis_jade_transf = (matrix_mixing_jade*pis_jade')';
            nis_jade = min(data_jade_adj); nis_jade_transf = (matrix_mixing_jade*nis_jade')';
            [simMM_ica_jade,orderMM_ica_jade,~,~,~] = func_topsis_mahalanobis_ica1(Nalt,data_mix,weights,pis_jade_transf,nis_jade_transf); % Jade
            
            sir(tt,kk,:) = [mean(calculate_SIR(data',data_fastica_adj')), mean(calculate_SIR(data',data_infomax_adj')), mean(calculate_SIR(data',data_jade_adj'))];  % Storing all the SIR
            metrics(tt,kk,:) = [metrics_fastica, metrics_infomax, metrics_jade]; % Storing all the Amari metrics
            
            %% Ordering and Kendall tau and Pearson coeficients
            orderEO_pos = orderEO; orderEO_pos(orderEO) = 1:Nalt; % Original (correct) ordering
            
            orderEM_pos = orderEM; orderEM_pos(orderEM) = 1:Nalt; % Mixed (Euclidean) ordering
            cc_EOEM = corr(simEM,simEO);
            kc_EOEM = corr(orderEM_pos,orderEO_pos,'type','Kendall');
            
            orderMM_pos = orderMM; orderMM_pos(orderMM) = 1:Nalt; % Mixed (Mahalanobis) ordering
            cc_EOMM = corr(simMM,simEO);
            kc_EOMM = corr(orderMM_pos,orderEO_pos,'type','Kendall');
            
            orderMM_utopic_pos = orderMM_utopic; orderMM_utopic_pos(orderMM_utopic) = 1:Nalt; % Utopic Mixed (Mahalanobis) ordering
            cc_EOMM_utopic = corr(simMM_utopic,simEO);
            kc_EOMM_utopic = corr(orderMM_utopic_pos,orderEO_pos,'type','Kendall');
            
            orderEI_pos_f = orderEI_f; orderEI_pos_f(orderEI_f) = 1:Nalt; % FastICA (Euclidean) ordering
            cc_EOEI_f = corr(simEI_f,simEO);
            kc_EOEI_f = corr(orderEI_pos_f,orderEO_pos,'type','Kendall');
            
            orderEI_pos_i = orderEI_i; orderEI_pos_i(orderEI_i) = 1:Nalt; % Infomax (Euclidean) ordering
            cc_EOEI_i = corr(simEI_i,simEO);
            kc_EOEI_i = corr(orderEI_pos_i,orderEO_pos,'type','Kendall');
            
            orderEI_pos_j = orderEI_j; orderEI_pos_j(orderEI_j) = 1:Nalt; % Jade (Euclidean) ordering
            cc_EOEI_j = corr(simEI_j,simEO);
            kc_EOEI_j = corr(orderEI_pos_j,orderEO_pos,'type','Kendall');
            
            orderMM_ica_fastica_pos = orderMM_ica_fastica; orderMM_ica_fastica_pos(orderMM_ica_fastica) = 1:Nalt; % Mixed (Mahalanobis+fastica) ordering
            cc_EOMM_ica_fastica = corr(simMM_ica_fastica,simEO);
            kc_EOMM_ica_fastica = corr(orderMM_ica_fastica_pos,orderEO_pos,'type','Kendall');
            
            orderMM_ica_infomax_pos = orderMM_ica_infomax; orderMM_ica_infomax_pos(orderMM_ica_infomax) = 1:Nalt; % Mixed (Mahalanobis+infomax) ordering
            cc_EOMM_ica_infomax = corr(simMM_ica_infomax,simEO);
            kc_EOMM_ica_infomax = corr(orderMM_ica_infomax_pos,orderEO_pos,'type','Kendall');
            
            orderMM_ica_jade_pos = orderMM_ica_jade; orderMM_ica_jade_pos(orderMM_ica_jade) = 1:Nalt; % Mixed (Mahalanobis+jade) ordering
            cc_EOMM_ica_jade = corr(simMM_ica_jade,simEO);
            kc_EOMM_ica_jade = corr(orderMM_ica_jade_pos,orderEO_pos,'type','Kendall');
            
            order_utopic_pos = order_utopic; order_utopic_pos(order_utopic) = 1:Nalt; % Utopic BSS ordering
            cc_utopic = corr(simEI_utopic,simEO);
            kc_utopic = corr(order_utopic_pos,orderEO_pos,'type','Kendall');
            
            coefcorr(tt,kk,:) = [cc_EOEM, cc_EOMM, cc_EOMM_utopic, cc_EOEI_f, cc_EOEI_i, cc_EOEI_j, cc_EOMM_ica_fastica, cc_EOMM_ica_infomax, cc_EOMM_ica_jade, cc_utopic]; % Storing all the Kendall tau distances
            kendallcorr(tt,kk,:) = [kc_EOEM, kc_EOMM, kc_EOMM_utopic, kc_EOEI_f, kc_EOEI_i, kc_EOEI_j, kc_EOMM_ica_fastica, kc_EOMM_ica_infomax, kc_EOMM_ica_jade, kc_utopic]; % Storing all the Kendall tau distances
            
            % Error first/last 20% of positions
            for ll=1:round(0.2*Nalt)
                error_first(tt,kk,1) = error_first(tt,kk,1) + abs(ll-orderEM_pos(orderEO(ll)));
                error_first(tt,kk,2) = error_first(tt,kk,2) + abs(ll-orderMM_pos(orderEO(ll)));
                error_first(tt,kk,3) = error_first(tt,kk,3) + abs(ll-orderMM_utopic_pos(orderEO(ll)));
                error_first(tt,kk,4) = error_first(tt,kk,4) + abs(ll-orderEI_pos_f(orderEO(ll)));
                error_first(tt,kk,5) = error_first(tt,kk,5) + abs(ll-orderEI_pos_i(orderEO(ll)));
                error_first(tt,kk,6) = error_first(tt,kk,6) + abs(ll-orderEI_pos_j(orderEO(ll)));
                error_first(tt,kk,7) = error_first(tt,kk,7) + abs(ll-orderMM_ica_fastica_pos(orderEO(ll)));
                error_first(tt,kk,8) = error_first(tt,kk,8) + abs(ll-orderMM_ica_infomax_pos(orderEO(ll)));
                error_first(tt,kk,9) = error_first(tt,kk,9) + abs(ll-orderMM_ica_jade_pos(orderEO(ll)));
                error_first(tt,kk,10) = error_first(tt,kk,10) + abs(ll-order_utopic_pos(orderEO(ll)));
                
                error_last(tt,kk,1) = error_last(tt,kk,1) + abs(Nalt+1-ll-orderEM_pos(orderEO(Nalt+1-ll)));
                error_last(tt,kk,2) = error_last(tt,kk,2) + abs(Nalt+1-ll-orderMM_pos(orderEO(Nalt+1-ll)));
                error_last(tt,kk,3) = error_last(tt,kk,3) + abs(Nalt+1-ll-orderMM_utopic_pos(orderEO(Nalt+1-ll)));
                error_last(tt,kk,4) = error_last(tt,kk,4) + abs(Nalt+1-ll-orderEI_pos_f(orderEO(Nalt+1-ll)));
                error_last(tt,kk,5) = error_last(tt,kk,5) + abs(Nalt+1-ll-orderEI_pos_i(orderEO(Nalt+1-ll)));
                error_last(tt,kk,6) = error_last(tt,kk,6) + abs(Nalt+1-ll-orderEI_pos_j(orderEO(Nalt+1-ll)));
                error_last(tt,kk,7) = error_last(tt,kk,7) + abs(Nalt+1-ll-orderMM_ica_fastica_pos(orderEO(Nalt+1-ll)));
                error_last(tt,kk,8) = error_last(tt,kk,8) + abs(Nalt+1-ll-orderMM_ica_infomax_pos(orderEO(Nalt+1-ll)));
                error_last(tt,kk,9) = error_last(tt,kk,9) + abs(Nalt+1-ll-orderMM_ica_jade_pos(orderEO(Nalt+1-ll)));
                error_last(tt,kk,10) = error_last(tt,kk,10) + abs(Nalt+1-ll-order_utopic_pos(orderEO(Nalt+1-ll)));
            end
            
            kk = kk + 1; % Update the counter
        end
    end
    tt
    save('resultado_simul_snr_2crit','coefcorr', 'kendallcorr', 'snr', 'metrics', 'error_first', 'error_last', 'tt', 'sir');
    
end

sir_mean = mean(sir);
metrics_mean = mean(metrics);
coefcorr_mean = mean(coefcorr);
kendallcorr_mean = mean(kendallcorr);
error_first_mean = mean(error_first);
error_last_mean = mean(error_last);
Nalt = 100;

figure; plot(0:length(snr)-1, sir_mean(1,:,1), '-r', 0:length(snr)-1, sir_mean(1,:,2), '-g', 0:length(snr)-1, sir_mean(1,:,3), '-b');
legend('SIR FastICA', 'SIR Infomax', 'SIR Jade');

figure; plot(0:length(snr)-1, metrics_mean(1,:,1), '-r', 0:length(snr)-1, metrics_mean(1,:,2), '-g', 0:length(snr)-1, metrics_mean(1,:,3), '-b');
legend('Amari metric FastICA', 'Amari metric Infomax', 'Amari metric Jade');

figure; plot(0:length(snr)-1, coefcorr_mean(1,:,1), '-y', 0:length(snr)-1, coefcorr_mean(1,:,2), '-g', 0:length(snr)-1, coefcorr_mean(1,:,3), '*-g', 0:length(snr)-1, coefcorr_mean(1,:,4), '-b', 0:length(snr)-1, coefcorr_mean(1,:,5), '-r', 0:length(snr)-1, coefcorr_mean(1,:,6), '-k', 0:length(snr)-1, coefcorr_mean(1,:,7), '*-b', 0:length(snr)-1, coefcorr_mean(1,:,8), '*-r', 0:length(snr)-1, coefcorr_mean(1,:,9), '*-k', 0:length(snr)-1, coefcorr_mean(1,:,10), '*-y');
legend('Euclidean', 'Mahalanobis', 'Mahalanobis utopic', 'Independent FastICA', 'Independent Infomax', 'Independent Jade', 'Mahalanobis FastICA', 'Mahalanobis Infomax', 'Mahalanobis Jade', 'Utopic');

figure; plot(0:length(snr)-1, kendallcorr_mean(1,:,1), '-y', 0:length(snr)-1, kendallcorr_mean(1,:,2), '-g', 0:length(snr)-1, kendallcorr_mean(1,:,3), '*-g', 0:length(snr)-1, kendallcorr_mean(1,:,4), '-b', 0:length(snr)-1, kendallcorr_mean(1,:,5), '-r', 0:length(snr)-1, kendallcorr_mean(1,:,6), '-k', 0:length(snr)-1, kendallcorr_mean(1,:,7), '*-b', 0:length(snr)-1, kendallcorr_mean(1,:,8), '*-r', 0:length(snr)-1, kendallcorr_mean(1,:,9), '*-k', 0:length(snr)-1, kendallcorr_mean(1,:,10), '*-y');
legend('Euclidean', 'Mahalanobis', 'Mahalanobis utopic', 'Independent FastICA', 'Independent Infomax', 'Independent Jade', 'Mahalanobis FastICA', 'Mahalanobis Infomax', 'Mahalanobis Jade', 'Utopic');

figure; plot(0:length(snr)-1, error_first_mean(1,:,1), '-y', 0:length(snr)-1, error_first_mean(1,:,2), '-g', 0:length(snr)-1, error_first_mean(1,:,3), '*-g', 0:length(snr)-1, error_first_mean(1,:,4), '-b', 0:length(snr)-1, error_first_mean(1,:,5), '-r', 0:length(snr)-1, error_first_mean(1,:,6), '-k', 0:length(snr)-1, error_first_mean(1,:,7), '*-b', 0:length(snr)-1, error_first_mean(1,:,8), '*-r', 0:length(snr)-1, error_first_mean(1,:,9), '*-k', 0:length(snr)-1, error_first_mean(1,:,10), '*-y');
legend('Euclidean', 'Mahalanobis', 'Mahalanobis utopic', 'Independent FastICA', 'Independent Infomax', 'Independent Jade', 'Mahalanobis FastICA', 'Mahalanobis Infomax', 'Mahalanobis Jade', 'Utopic');

figure; plot(0:length(snr)-1, error_last_mean(1,:,1), '-y', 0:length(snr)-1, error_last_mean(1,:,2), '-g', 0:length(snr)-1, error_last_mean(1,:,3), '*-g', 0:length(snr)-1, error_last_mean(1,:,4), '-b', 0:length(snr)-1, error_last_mean(1,:,5), '-r', 0:length(snr)-1, error_last_mean(1,:,6), '-k', 0:length(snr)-1, error_last_mean(1,:,7), '*-b', 0:length(snr)-1, error_last_mean(1,:,8), '*-r', 0:length(snr)-1, error_last_mean(1,:,9), '*-k', 0:length(snr)-1, error_last_mean(1,:,10), '*-y');
legend('Euclidean', 'Mahalanobis', 'Mahalanobis utopic', 'Independent FastICA', 'Independent Infomax', 'Independent Jade', 'Mahalanobis FastICA', 'Mahalanobis Infomax', 'Mahalanobis Jade', 'Utopic');


% Figures separating the proposed approaches
figure; plot(0:length(snr)-1, sir_mean(1,:,1), '-r', 0:length(snr)-1, sir_mean(1,:,3), '-b');
legend('FastICA', 'JADE'); xlabel('SNR (dB)'); ylabel('SIR (dB)'); 

figure; plot(0:length(snr)-1, metrics_mean(1,:,1), '-r', 0:length(snr)-1, metrics_mean(1,:,3), '-b');
legend('FastICA', 'JADE'); xlabel('SNR (dB)'); ylabel('$\theta$');

figure; plot(0:length(snr)-1, coefcorr_mean(1,:,1), '-y', 0:length(snr)-1, coefcorr_mean(1,:,2), '-g', 0:length(snr)-1, coefcorr_mean(1,:,4), '-b', 0:length(snr)-1, coefcorr_mean(1,:,6), '-k', 0:length(snr)-1, coefcorr_mean(1,:,10), '*-y');
legend('TOPSIS', 'TOPSIS-M', 'ICA-TOPSIS (FastICA)', 'ICA-TOPSIS (JADE)', 'Utopic ICA-TOPSIS'); xlabel('SNR (dB)'); ylabel('$\rho$');
figure; plot(0:length(snr)-1, coefcorr_mean(1,:,1), '-y', 0:length(snr)-1, coefcorr_mean(1,:,2), '-g', 0:length(snr)-1, coefcorr_mean(1,:,7), '*-b', 0:length(snr)-1, coefcorr_mean(1,:,9), '*-k', 0:length(snr)-1, coefcorr_mean(1,:,3), '*-g');
legend('TOPSIS', 'TOPSIS-M', 'ICA-TOPSIS-M (FastICA)', 'ICA-TOPSIS-M (JADE)', 'Utopic ICA-TOPSIS-M'); xlabel('SNR (dB)'); ylabel('$\rho$');

figure; plot(0:length(snr)-1, kendallcorr_mean(1,:,1), '-y', 0:length(snr)-1, kendallcorr_mean(1,:,2), '-g', 0:length(snr)-1, kendallcorr_mean(1,:,4), '-b', 0:length(snr)-1, kendallcorr_mean(1,:,6), '-k', 0:length(snr)-1, kendallcorr_mean(1,:,10), '*-y');
legend('TOPSIS', 'TOPSIS-M', 'ICA-TOPSIS (FastICA)', 'ICA-TOPSIS (JADE)', 'Utopic ICA-TOPSIS'); xlabel('SNR (dB)'); ylabel('$\tau$');
figure; plot(0:length(snr)-1, kendallcorr_mean(1,:,1), '-y', 0:length(snr)-1, kendallcorr_mean(1,:,2), '-g', 0:length(snr)-1, kendallcorr_mean(1,:,7), '*-b', 0:length(snr)-1, kendallcorr_mean(1,:,9), '*-k', 0:length(snr)-1, kendallcorr_mean(1,:,3), '*-g');
legend('TOPSIS', 'TOPSIS-M', 'ICA-TOPSIS-M (FastICA)', 'ICA-TOPSIS-M (JADE)', 'Utopic ICA-TOPSIS-M'); xlabel('SNR (dB)'); ylabel('$\tau$');

error_first_mean = error_first_mean/round(0.2*Nalt);
figure; plot(0:length(snr)-1, error_first_mean(1,:,1), '-y', 0:length(snr)-1, error_first_mean(1,:,2), '-g', 0:length(snr)-1, error_first_mean(1,:,4), '-b', 0:length(snr)-1, error_first_mean(1,:,6), '-k', 0:length(snr)-1, error_first_mean(1,:,10), '*-y');
legend('TOPSIS', 'TOPSIS-M', 'ICA-TOPSIS (FastICA)', 'ICA-TOPSIS (JADE)', 'Utopic ICA-TOPSIS'); xlabel('SNR (dB)'); ylabel('$\varepsilon$');
figure; plot(0:length(snr)-1, error_first_mean(1,:,1), '-y', 0:length(snr)-1, error_first_mean(1,:,2), '-g', 0:length(snr)-1, error_first_mean(1,:,7), '*-b', 0:length(snr)-1, error_first_mean(1,:,9), '*-k', 0:length(snr)-1, error_first_mean(1,:,3), '*-g');
legend('TOPSIS', 'TOPSIS-M', 'ICA-TOPSIS-M (FastICA)', 'ICA-TOPSIS-M (JADE)', 'Utopic ICA-TOPSIS-M'); xlabel('SNR (dB)'); ylabel('$\varepsilon$');

error_last_mean = error_last_mean/round(0.2*Nalt);
figure; plot(0:length(snr)-1, error_last_mean(1,:,1), '-y', 0:length(snr)-1, error_last_mean(1,:,2), '-g', 0:length(snr)-1, error_last_mean(1,:,4), '-b', 0:length(snr)-1, error_last_mean(1,:,6), '-k', 0:length(snr)-1, error_last_mean(1,:,10), '*-y');
legend('TOPSIS', 'TOPSIS-M', 'ICA-TOPSIS (FastICA)', 'ICA-TOPSIS (JADE)', 'Utopic ICA-TOPSIS'); xlabel('SNR (dB)'); ylabel('$\varepsilon$');
figure; plot(0:length(snr)-1, error_last_mean(1,:,1), '-y', 0:length(snr)-1, error_last_mean(1,:,2), '-g', 0:length(snr)-1, error_last_mean(1,:,7), '*-b', 0:length(snr)-1, error_last_mean(1,:,9), '*-k', 0:length(snr)-1, error_last_mean(1,:,3), '*-g');
legend('TOPSIS', 'TOPSIS-M', 'ICA-TOPSIS-M (FastICA)', 'ICA-TOPSIS-M (JADE)', 'Utopic ICA-TOPSIS-M'); xlabel('SNR (dB)'); ylabel('$\varepsilon$');
