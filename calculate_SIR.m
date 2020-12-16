% Calculate Signal-to-interference ratio (SIR) considering the real sources and the estimated ones
% Inputs
%       S: real sources
%       SE: estimated sources
% Outputs
%       SIR: vector containing SIR for each pair source/estimated signal
%       SIR_matrix: matrix containing SIRs for all combinations


function [SIR, SIR_matrix] = calculate_SIR(S,SE)

[Ns Nd] = size(S);

for ii=1:Ns
    S(ii,:) = (S(ii,:)-mean(S(ii,:)))/std(S(ii,:));
    SE(ii,:) = (SE(ii,:)-mean(SE(ii,:)))/std(SE(ii,:));
end

for ii = 1:Ns
    for jj = 1:Ns
        K = mean(S(ii,:).*SE(jj,:))/mean(SE(jj,:).^2); % Compensating scaling ambiguitities
        SE_temp(jj,:) = K*SE(jj,:);
        SIR_matrix(ii,jj) = 10*log10( (mean(S(ii,:).^2))/ mean((S(ii,:)-SE_temp(jj,:)).^2));
    end
        
    SIR(ii) = max(SIR_matrix(ii,:));
    
end