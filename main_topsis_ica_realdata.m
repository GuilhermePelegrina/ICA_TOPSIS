%% Experiment with real data
% Use JADE as ICA technique

clear all; close all; clc;

%% Load data
load data_real_paper.mat;

%% Paramenters
[Nalt, Ncrit] = size(data_mix); % # of alternatives and # of criteria
weights = ones(1,size(data_mix,2))*(1/size(data_mix,2)); % Equal weights
% data_mix = (data_mix - repmat(min(data_mix),Nalt,1))./(repmat(max(data_mix),Nalt,1)-repmat(min(data_mix),Nalt,1));

%% Data visualization
figure; plot(data_mix(:,1), data_mix(:,2),'.k'); xlabel('$\mathcal{C}_1$: Forest area (\% of land area)'); ylabel('$\mathcal{C}_2$: GNI per capita (current US\$)');
figure; plot(data_mix(:,1), data_mix(:,3),'.k'); xlabel('$\mathcal{C}_1$: Forest area (\% of land area)'); ylabel('$\mathcal{C}_3$: Life expectancy at birth (years)');
figure; plot(data_mix(:,2), data_mix(:,3),'.k'); xlabel('$\mathcal{C}_2$: GNI per capita (current US\$)'); ylabel('$\mathcal{C}_3$: Life expectancy at birth (years)');

%% Data characteristics
MaxMin = [max(data_mix); min(data_mix)]; % Maximum and minimum value in the observed criteria
data_corr = corr(data_mix); % Correlation coefficient among the observed criteria

%% TOPSIS in mixed data
[simEM,orderEM,~,~,~] = func_topsis_euclidean1(Nalt,data_mix,weights);

%% ICA and ICA-TOPSIS in independent data
B = jadeR(data_mix',Ncrit);
data_jade = B*data_mix';
data_jade = data_jade';
[data_jade_adj,matrix_mixing_jade] = func_bss_correct(Nalt,Ncrit,inv(B),data_jade);
[simEI_j,orderEI_j,~,~,~] = func_topsis_euclidean1(Nalt,data_jade_adj,weights);

%% TOPSIS-M in mixed data
[simMM,orderMM,~,~,~] = func_topsis_mahalanobis1(Nalt,data_mix,weights);

%% ICA-TOPSIS-M in mixed data
pis_jade = max(data_jade_adj);
nis_jade = min(data_jade_adj);
pis_jade_transf = (matrix_mixing_jade*pis_jade')';
nis_jade_transf = (matrix_mixing_jade*nis_jade')';
[simMM_ica_jade,orderMM_ica_jade,~,~,~] = func_topsis_mahalanobis_ica1(Nalt,data_mix,weights,pis_jade_transf,nis_jade_transf);

%% Monotonicity verification
vm_EM = verify_monotonicity(data_mix,simEM);
vm_MM = verify_monotonicity(data_mix,simMM);
vm_EI_j = verify_monotonicity(data_mix,simEI_j);
vm_MM_ica_jade = verify_monotonicity(data_mix,simMM_ica_jade);

%% Correlation coefficient and Kendall tau between the approaches
matr_coef = (1/2)*eye(4);
matr_coef(1,2) = corr(simEM,simMM); matr_coef(1,3) = corr(simEM,simEI_j); matr_coef(1,4) = corr(simEM,simMM_ica_jade);
matr_coef(2,3) = corr(simMM,simEI_j); matr_coef(2,4) = corr(simMM,simMM_ica_jade);
matr_coef(3,4) = corr(simEI_j,simMM_ica_jade);
matr_coef = matr_coef + matr_coef';

matr_kend = (1/2)*eye(4);
matr_kend(1,2) = corr(orderEM,orderMM,'type','Kendall'); matr_kend(1,3) = corr(orderEM,orderEI_j,'type','Kendall'); matr_kend(1,4) = corr(orderEM,orderMM_ica_jade,'type','Kendall');
matr_kend(2,3) = corr(orderMM,orderEI_j,'type','Kendall'); matr_kend(2,4) = corr(orderMM,orderMM_ica_jade,'type','Kendall');
matr_kend(3,4) = corr(orderEI_j,orderMM_ica_jade,'type','Kendall');
matr_kend = matr_kend + matr_kend';

%% Comparison between the rankings (first 10 alternatives)
% Table columns: Alternatives - Evaluations(C1-C2-C3) - Positions(Em-MM-EI_j-MM_ica_jade)
table(:,1) = sort(unique([orderEM(1:10); orderMM(1:10); orderEI_j(1:10); orderMM_ica_jade(1:10)])); % Alternatives
table(:,2:4) = data_mix(table(:,1),:); % Evaluations(C1-C2-C3)
for ii=1:16
    table(ii,5) = find(orderEM==table(ii,1));
    table(ii,6) = find(orderMM==table(ii,1));
    table(ii,7) = find(orderEI_j==table(ii,1));
    table(ii,8) = find(orderMM_ica_jade==table(ii,1));
end
    