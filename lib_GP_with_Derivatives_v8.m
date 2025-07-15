function [YdY_star Y_star dY_star YdY_star_cov] = lib_GP_with_Derivatives (X_star, Observations, hyperparams)
% ....................... check the ^2 factors in the kernel Jan-2023 [DC]
% .................................................1st time initialization 
% >> lib_GP_with_Derivatives ([] , Observations, hyperparams)
% - Observations    structure with fields
%       .X     D x S locations (each in R^D) [x_1 ... x_S]
%       .Y     1 x S scalar observations     [y_1 ... y_S]
%       .dX    D x M locations (each in R^D) [dx_1 .. dx_M]
%       .dY    D x M scalar observations     [dy_1 ... dy_M]
% - hyperparams 
%       [dx1, dx2, ..., dxD, std_kernel, std_noise]'
% o GP structure with function handles
%       .Y       = @(x) with x Dx1
%       .grad_Y  = @(x) with x Dx1   WARNING: NOT FULLY TESTED
%
% .......................................routinely called with one arument 
% >> [Y_star dY_star YdY_star_cov] = lib_Gaussian_Process_Regression (X_star) 
% - X_star      [D x Nstar] D-vector at Nstar locations, point of regression
% returns
% o YdY_star    [1+D x N_star]
% o Y_star      [1 x Nstar] scalar    regressions @ each X_star location 
% o dY_star     [D x Nstar] gradient  regressions @ each X_star location 
% o Y_star_cov  covariance of [Y_star(:); dY_star(:)]  *** TO-BE-TESTED ***

persistent maha kernel grad_kernel Hess_kernel alpha L X dX Y dY D S M ;

    if nargin>1                  %% meant to happen only at initialization
        disp(".............. initializing the GP library")
        X = Observations.X;
        dX = Observations.dX;

        Y = Observations.Y;
        dY = Observations.dY;
        Y_all = [Y(:); dY(:)];      %% vertical vector (S+DM)x 1
        
        D = max( size(X, 1), size(dX, 1)); %% dimensionality of X variables
        S = size(Y, 2);
        M = size(dY, 2);
        
        if nargin>2         %% if 'hyp' is given as argument
            r = hyperparams(1:D);
            stdDev_k = hyperparams(end-1);
            stdDev_n = hyperparams(end);
        else
            r = 0.1 * ones(1,D);  %% length-scales for all of D dimensions
            stdDev_k = 1;
            stdDev_n = 0.1;
        end
            syms x_a x_b [D 1] real;
            maha =@(x, W) x' * W * x;
%             sym.kernel = stdDev_k^2 * exp(-1/2 * maha(x_a-x_b, diag( 1./r )^2 )); %% ................ check the ^2 factors
%             sym.kernel = stdDev_k^2 * exp(-1/2 * maha(x_a-x_b, diag( 1./r ) )); %% ................ check the ^2 factors
%             sym.kernel = exp(-1/2 * maha(x_a-x_b, diag( 1./r )^2 )); %% 1/31/2023
            sym.kernel = stdDev_k^2 * exp(-1/2 * maha(x_a-x_b, diag( 1./r )^2 )); %% 4/feb/2023
            sym.grad_kernel = jacobian(sym.kernel, x_a)';
                % NOTE: jacobian(sym.Kernel, x_a)' + jacobian(sym.Kernel, x_b)' ==0

            sym.Hess_kernel = - hessian(sym.kernel, x_a);               %%%% ............................CHECK minus sign
                % NOTE: jacobian( jacobian(sym.Kernel, x_a)' , x_b)+ hessian(sym.Kernel, x_a) == 0
                % NOTE: hessian(sym.Kernel, x_a) - hessian(sym.Kernel, x_b) == 0

            kernel      = matlabFunction (sym.kernel, 'Vars', {x_a, x_b});
            grad_kernel = matlabFunction (sym.grad_kernel, 'Vars', {x_a, x_b});
            Hess_kernel = matlabFunction (sym.Hess_kernel, 'Vars', {x_a, x_b});        
   
        Kii = [ BuildCovMat(kernel, X, X), BuildCovMat(grad_kernel, dX, X)';
                BuildCovMat(grad_kernel, dX, X), BuildCovMat(Hess_kernel, dX, dX)];
                   
        L = chol (Kii + stdDev_n^2 * eye(S+D*M), 'lower');    %% Choleski lower triangular decomposition      %% CHECK ^2
        alpha = L'\(L\Y_all);  %% as opposed to   Koi * G * Y_all 

        disp (".............. generated and inverted Kii of size: " + size(Kii,1) + "x" + size(Kii,2) )
        
        if nargout>0
            disp('*** WARNING: GP functionality NOT fully tested') ;
            aux_row = @(x_str) exp(-1/2 * sum((x_str-X).*(x_str-X).*(1./(r.^2))) );
            GP.Y = @(x_str ) stdDev_k^2 * aux_row(x_str) * alpha ; %% 5/feb/2023
            GP.grad_Y = @(x_str ) - stdDev_k^2 * (((x_str-X).*(1./(r.^2))) .* aux_row(x_str) ) * alpha ; %% 5/feb/2023
            YdY_star = GP;
        end
        return;     %% only the first time is called with >1 inputs, and the firts input can be ignored
        
    end
  
    Koi = [ BuildCovMat(kernel, X_star, X),    - BuildCovMat(@(a,b)grad_kernel(a,b)', X_star, dX);  %% NEEDED to transpose the grad_kernel, $- \nabla^T \kappa$
            BuildCovMat(grad_kernel, X_star, X),  BuildCovMat(Hess_kernel, X_star, dX)];
  
    N_star = size(X_star, 2);   %% X_star is D x N_star (typically N_star=1)
    
    YdY_star = Koi * alpha;    %% as opposed to   Koi * G * Yi' 
    Y_star = YdY_star(1:N_star)';
    dY_star = reshape(YdY_star(N_star+1:end), size(X_star));

    if nargout>3
        V = L\(Koi');
        Koo = blkdiag( BuildCovMat(kernel, X_star, X_star), BuildCovMat(Hess_kernel, X_star, X_star));
        YdY_star_cov = Koo - V'*V; %% as opposed to   Koo - Kio * G * Koi  
    end

end


%%
function Kab = BuildCovMat (somekernel, A, B)

    % if size(A,1)>size(A,2) || size(B,1)>size(B,2)
    %     disp('WARNING: datasets are expected to be D x n matrices, with D<n')
    % end

    KKab = cell(size(A,2), size(B,2));
    for a=1 : size(A,2)
        for b=1 : size(B,2)
            KKab{a,b} = somekernel(A(:,a), B(:,b) );
        end
    end
    Kab = cell2mat(KKab);
end

