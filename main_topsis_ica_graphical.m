%% Graphical interpretation of ICA TOPSIS analysis
% Comparison considering the following approaches:
%   TOPSIS-E in latent variables
%   TOPSIS-E in mixed data
%   TOPSIS-M in mixed data
%   ICA-TOPSIS (Jade) in retrieved sources
%   ICA-TOPSIS-M (Jade) in mixed data (with transformed PIS and NIS)
% Apply vector normalization in decision data
% Mixing matrix given by [1 0.7; -0.25 1]
% Weighted evaluations (...euclidean/mahalanobis1) or Weighted distances (...euclidean/mahalanobis2)
% Number of alternatives: 100
% Without additive noise
% Equal weights

clear all; close all; clc;

%% Dataset and mixing matrix
load data_graphical_analysis; % Data and mixing matrix
Mmixing = [1 0.7; -0.25 1];
[Nalt, Ncrit] = size(data); % # of alternatives and # of criteria
Nalt = 200;
data = rand(Nalt,Ncrit);

% data(:,1) = rand(Nalt,1); data(:,2) = 1000*rand(Nalt,1);

%% Criteria weights
weights = ones(1,size(data,2))*(1/size(data,2)); % Equal weights

%% TOPSIS (Euclidean) in original data
[simEO,orderEO,dataEO,pisEO,nisEO] = func_topsis_euclidean1(Nalt,data,weights); % euclidean - original data

%% Mixed data
data_mix = (Mmixing*data')'; % Generate the mixed data

%% TOPSIS (Eucledian) in mixed data
[simEM,orderEM,dataEM,pisEM,nisEM] = func_topsis_euclidean1(Nalt,data_mix,weights); % euclidean - mixed data
pisEM_corr = weights.*((Mmixing*max(data)')'./sqrt(sum(data_mix.^2)));
nisEM_corr = weights.*((Mmixing*min(data)')'./sqrt(sum(data_mix.^2)));

%% Extracting independent data (ICA by Jade)
B = jadeR(data_mix',Ncrit);
data_jade = B*data_mix';
data_jade = data_jade';
[data_jade_adj,matrix_mixing_jade] = func_bss_correct(Nalt,Ncrit,inv(B),data_jade);

%% TOPSIS (Euclidean) in independent data
[simEI_j,orderEI_j,dataEI_j,pisEI_j,nisEI_j] = func_topsis_euclidean1(Nalt,data_jade_adj,weights); % euclidean - Jade
pisEI_j_corr = weights.*((inv(matrix_mixing_jade)*Mmixing*max(data)')'./sqrt(sum(data_jade_adj.^2)));
nisEI_j_corr = weights.*((inv(matrix_mixing_jade)*Mmixing*min(data)')'./sqrt(sum(data_jade_adj.^2)));

%% TOPSIS (Mahalanobis) in mixed data
Dweights = diag(weights); % Defining diagonal matrix of weights
norm_rat = data_mix./repmat(sqrt(sum(data_mix(1:Nalt,:).^2)),Nalt,1); % Calculating normalized ratings
C_inv = inv(cov(norm_rat(1:Nalt,:))); % Calculating the inverse of the covariance matrix
L = chol(inv(C_inv), 'lower'); % Cholesky factorization

[simMM,orderMM,dataMM,pisMM,nisMM] = func_topsis_mahalanobis1(Nalt,data_mix,weights); % Mahalanobis - mixed data
dataMM2 = (inv(L)*Dweights*norm_rat')'; % Transformed data
pisMM2 = (inv(L)*Dweights*max(norm_rat(1:Nalt,:))')'; % Transformed PIS
nisMM2 = (inv(L)*Dweights*min(norm_rat(1:Nalt,:))')'; % Transformed NIS
pisMM2_corr = (inv(L)*Dweights*((Mmixing*max(data)')'./sqrt(sum(data_mix.^2)))')';
nisMM2_corr = (inv(L)*Dweights*((Mmixing*min(data)')'./sqrt(sum(data_mix.^2)))')';

%% TOPSIS (Mahalanobis) in mixed data with tronsformed PIS and NIS (ICA by Jade)
pis_jade = max(data_jade_adj);
nis_jade = min(data_jade_adj);
pis_jade_transf = (matrix_mixing_jade*pis_jade')';
nis_jade_transf = (matrix_mixing_jade*nis_jade')';
[simMM_ica_jade,orderMM_ica_jade,dataMM_ica_jade,pisMM_ica_jade,nisMM_ica_jade] = func_topsis_mahalanobis_ica1(Nalt,data_mix,weights,pis_jade_transf,nis_jade_transf); % Mahalanobis - mixed data
dataMM_ica_jade2 = (inv(L)*Dweights*norm_rat')';
pisMM_ica_jade2 = (inv(L)*Dweights*(pis_jade_transf./sqrt(sum(data_mix(1:Nalt,:).^2)))')';
nisMM_ica_jade2 = (inv(L)*Dweights*(nis_jade_transf./sqrt(sum(data_mix(1:Nalt,:).^2)))')';
pisMM_ica_jade2_corr = (inv(L)*Dweights*((Mmixing*max(data)')'./sqrt(sum(data_mix.^2)))')';
nisMM_ica_jade2_corr = (inv(L)*Dweights*((Mmixing*min(data)')'./sqrt(sum(data_mix.^2)))')';

%% Figures

% figure; subplot(1,3,1); plot(dataEO(:,1),dataEO(:,2),'k.', pisEO(1), pisEO(2), 'b*', nisEO(1), nisEO(2), 'r*');
% legend('Alternatives', 'PIA', 'NIA'); title('TOPSIS-E in latent variables');
% subplot(1,3,2); plot(dataEM(:,1),dataEM(:,2),'k.', pisEM(1), pisEM(2), 'b*', nisEM(1), nisEM(2), 'r*', pisEM_corr(1), pisEM_corr(2), 'bo', nisEM_corr(1), nisEM_corr(2), 'ro');
% legend('Alternatives', 'PIA', 'NIA', 'Correct PIA', 'Correct NIA'); title('TOPSIS-E in decision (mixed) data');
% subplot(1,3,3); plot(dataMM2(:,1),dataMM2(:,2),'k.', pisMM2(1), pisMM2(2), 'b*', nisMM2(1), nisMM2(2), 'r*', pisMM2_corr(1), pisMM2_corr(2), 'bo', nisMM2_corr(1), nisMM2_corr(2), 'ro');
% legend('Alternatives', 'PIA', 'NIA', 'Correct PIA', 'Correct NIA'); title('TOPSIS-M in decision (mixed) data');
% 
% figure; subplot(1,3,1); plot(dataEO(:,1),dataEO(:,2),'k.', pisEO(1), pisEO(2), 'b*', nisEO(1), nisEO(2), 'r*');
% legend('Alternatives', 'PIA', 'NIA'); title('TOPSIS-E in latent variables');
% subplot(1,3,2); plot(dataEI_j(:,1),dataEI_j(:,2),'k.', pisEI_j(1), pisEI_j(2), 'b*', nisEI_j(1), nisEI_j(2), 'r*', pisEI_j_corr(1), pisEI_j_corr(2), 'bo', nisEI_j_corr(1), nisEI_j_corr(2), 'ro');
% legend('Alternatives', 'PIA', 'NIA', 'Correct PIA', 'Correct NIA'); title('ICA-TOPSIS in decision (mixed) data');
% subplot(1,3,3); plot(dataMM_ica_jade2(:,1),dataMM_ica_jade2(:,2),'k.', pisMM_ica_jade2(1), pisMM_ica_jade2(2), 'b*', nisMM_ica_jade2(1), nisMM_ica_jade2(2), 'r*', pisMM_ica_jade2_corr(1), pisMM_ica_jade2_corr(2), 'bo', nisMM_ica_jade2_corr(1), nisMM_ica_jade2_corr(2), 'ro');
% legend('Alternatives', 'PIA', 'NIA', 'Correct PIA', 'Correct NIA'); title('ICA-TOPSIS-M in decision (mixed) data');

figure; plot(dataEO(:,1),dataEO(:,2),'k.', pisEO(1), pisEO(2), 'b*', nisEO(1), nisEO(2), 'r*');
ylabel('Latent variable 1'); xlabel('Latent variable 2'); legend('Alternatives', 'Correct PIA', 'Correct NIA');
figure; plot(dataEM(:,1),dataEM(:,2),'k.', pisEM(1), pisEM(2), 'b*', nisEM(1), nisEM(2), 'r*', pisEM_corr(1), pisEM_corr(2), 'bo', nisEM_corr(1), nisEM_corr(2), 'ro');
ylabel('Mixed data 1'); xlabel('Mixed data 2');; legend('Alternatives', 'PIA', 'NIA', 'Correct PIA', 'Correct NIA');
figure; plot(dataMM2(:,1),dataMM2(:,2),'k.', pisMM2(1), pisMM2(2), 'b*', nisMM2(1), nisMM2(2), 'r*', pisMM2_corr(1), pisMM2_corr(2), 'bo', nisMM2_corr(1), nisMM2_corr(2), 'ro');
ylabel('Uncorrelated data 1'); xlabel('Uncorrelated data 2'); legend('Alternatives', 'PIA', 'NIA', 'Correct PIA', 'Correct NIA');
figure; plot(dataEI_j(:,1),dataEI_j(:,2),'k.', pisEI_j(1), pisEI_j(2), 'b*', nisEI_j(1), nisEI_j(2), 'r*', pisEI_j_corr(1), pisEI_j_corr(2), 'bo', nisEI_j_corr(1), nisEI_j_corr(2), 'ro');
ylabel('Independent data 1'); xlabel('Independent data 2'); legend('Alternatives', 'PIA', 'NIA', 'Correct PIA', 'Correct NIA');
figure; plot(dataMM_ica_jade2(:,1),dataMM_ica_jade2(:,2),'k.', pisMM_ica_jade2(1), pisMM_ica_jade2(2), 'b*', nisMM_ica_jade2(1), nisMM_ica_jade2(2), 'r*', pisMM_ica_jade2_corr(1), pisMM_ica_jade2_corr(2), 'bo', nisMM_ica_jade2_corr(1), nisMM_ica_jade2_corr(2), 'ro');
ylabel('Uncorrelated data 1'); xlabel('Uncorrelated data 2'); legend('Alternatives', 'PIA', 'NIA', 'Correct PIA', 'Correct NIA');
