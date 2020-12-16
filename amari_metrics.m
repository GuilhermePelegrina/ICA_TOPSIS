function [metrics] = amari_metrics(matrix_sep,Ncrit,Mmixing)

matrix_metrics = matrix_sep*Mmixing;
for ii=1:Ncrit
    metrics_aux1(ii) = (sum(abs(matrix_metrics(ii,:)))/max(abs(matrix_metrics(ii,:))))-1;
    metrics_aux2(ii) = (sum(abs(matrix_metrics(:,ii)))/max(abs(matrix_metrics(:,ii))))-1;
end
metrics = (1/(2*Ncrit*(Ncrit-1)))*(sum(metrics_aux1)+sum(metrics_aux1));

end

