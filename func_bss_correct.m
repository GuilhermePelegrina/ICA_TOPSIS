function [data_aux,matrix_mixing_aux] = func_bss_correct(Nalt,Ncrit,matrix_mixing,data)

data_aux = zeros(Nalt,Ncrit); % Correcting sinal ambiguity provided by ICA (hypothesis)
matrix_mixing_aux = zeros(Ncrit,Ncrit);
matrix_mixing_cop = matrix_mixing;
cont = 1;
indexes = [1:Ncrit];
while size(matrix_mixing_cop,2) > 0
    [~,index] = max(abs(matrix_mixing_cop(cont,:)));
    matrix_mixing_aux(:,cont) = sign(matrix_mixing_cop(cont,index))*matrix_mixing(:,indexes(index));
    data_aux(:,cont) = sign(matrix_mixing_cop(cont,index))*data(:,indexes(index));
    cont = cont + 1;
    indexes(index) = [];
    matrix_mixing_cop(:,index) = [];
end


end

