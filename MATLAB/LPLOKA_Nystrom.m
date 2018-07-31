function F = LPLOKA_Nystrom(S)
    % S: (m x n) Sequence similarity matrix, normalized as, for example,
    % exp(-eval/sigma). m is the total number of sequences (database size),
    % and n < m the number of sequences of interest (such as those from
    % SCOP-40)
    % F: (m x n) is the output from nystrom, such as F*F' = X, in which X
    % is the pairwise sequence similarity m x m, that we wish to
    % approximate

    [~, n] = size(S);
    
    % making W block symmetric
    S(1:n,:) = (S(1:n,:)+S(1:n,:)')/2;

    % making W block positive definite (Gershgorin circle theorem)
    for i=1:n
        S(i,i) = sum(abs(S(:,i)));
    end

    %% Nystrom
    % for reference:
    %   W = S(1:n,:);
    %   C = S;
    
    % eigen decomposition of W
    [U,A] = eig(full(S(1:n,:)));
    eig_a = diag(A);
    clear A;

    % find Sigma_k pseudo-inverse, and take square root
    Sigk = sparse(1:n, 1:n, eig_a.^(-0.5));
    clear eig_a;
    
    F = S*U*Sigk;
    clear U Sigk
    
    %% normalize F
    s = abs(F*sum(F)');
    s(s==0) = 1;
    F = bsxfun(@rdivide, F, sqrt(s));
    
end

