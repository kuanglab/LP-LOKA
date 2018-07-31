function Fhat = LPLOKA(F, F0, varargin)
    % F: (m x n) is the output from Nystrom
    % F0: (m x n) initialization (usually same as S, the input used by
    % Nystrom)
    % Optional Arguments:
    %   'Alpha': Label propagation hyper-parameter (default 0.1)
    %   'MaxIter': Maximum number of iterations (default 20)
    %   'Tol': Maximum element-wise difference between consecutive
    %   iterations, such as max_ij(|F_old - F_new|) < tol to stop execution
    %   (default 1e-9)

    alpha = 0.1;
    maxIter = 20;
    tol = 1e-9;

    v = 1;
    while v < numel(varargin)
      switch varargin{v}
      case 'Alpha'
        v = v+1;
        alpha = varargin{v};
        assert(alpha > 0 && alpha < 1);
      case 'MaxIter'
        v = v+1;
        maxIter = varargin{v};
        assert(maxIter > 0);
      case 'Tol'
        v = v+1;
        tol = varargin{v};
      end
      v = v+1;
    end

    [m, n] = size(F0);
    F0 = bsxfun(@rdivide, F0, sum(F0));

    %% initialization
    Fhat = ones(m,n)/n;

    %% main iteration
    fprintf('Starting LPLOKA (alpha=%.4f)\n', alpha);
    for iter = 1:maxIter
        fprintf('Iteration: %d/%d ... ', iter, maxIter);
        Fold = Fhat;

        % update
        Fhat = alpha*(F*(F'*Fold)) + (1-alpha)*F0;

        diff = max(max(abs(Fhat-Fold)));
        fprintf('Convergence: %.2x\n', diff);
        if diff < tol
            break;
        end
    end

end
