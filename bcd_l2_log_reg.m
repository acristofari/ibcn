% Block Coordinate Descent method with greedy selection for l2-regularized logistic regression inspired by
% "Tseng, P., & Yun, S. (2009). A coordinate gradient descent method for nonsmooth separable minimization. Mathematical Programming, 117, 387-423".
% 
% This code is used in the paper "A. Cristofari. Block cubic Newton with greedy selection. arXiv:2407.18150"
% for comparison with a proposed algorithm.
% 
% Author: Andrea Cristofari (andrea.cristofari@uniroma2.it)
% Last update of this file: July 26th, 2024
% 
% Licensing:
% This file is part of IBCN.
% IBCN is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% IBCN is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% You should have received a copy of the GNU General Public License
% along with IBCN. If not, see <http://www.gnu.org/licenses/>.
% 
% Copyright 2024 Andrea Cristofari.

function [x,it,flag,f_vec,g_vec] = bcd_l2_log_reg(A,b,x,lambda,options)
    
    if (~isnumeric(A) || ~isreal(A) || ~ismatrix(A))
        error('the first input must be a real matrix');
    end
    A = [A ones(size(A,1),1)];
    [l,n] = size(A);
    if (~isnumeric(b) || ~isreal(b) || ~iscolumn(b))
        error('the second input must be a real column vector');
    end
    if (size(b,1) ~= l)
        error('the dimension of A and b must agree');
    end
    if (~isnumeric(x) || ~isreal(x) || ~iscolumn(x))
        error('the third input must be a real column vector');
    end
    if (size(x,1) ~= n)
        error('the dimension of A and x must agree');
    end
    if (~isnumeric(lambda) || ~isreal(lambda) || ~isscalar(lambda) || lambda<0e0)
        error('lambda must be a non-negative real number');
    end
    
    % set options (if any, default values otherwise)
    block_dim = 1;
    eps = 1e-5;
    max_it = 10000;
    verbosity = false;
    if (nargin == 5)
        if (~isstruct(options) || ~isscalar(options))
            error('the sixth input (which is optional) must be a structure');
        end
        opts_field = fieldnames(options);
        for i = 1:length(opts_field)
            switch char(opts_field(i))
                case 'block_dim'
                    block_dim = options.block_dim;
                    if (~isnumeric(block_dim) || ~isreal(block_dim) || ~isscalar(block_dim) || block_dim<1e0 || block_dim>n)
                       error(['''block_dim'' must be a real number >= 1 and <= ' num2str(n)]);
                    end
                    block_dim = round(block_dim);
                case 'eps_opt'
                    eps = options.eps_opt;
                    if (~isnumeric(eps) || ~isreal(eps) || ~isscalar(eps) || eps<0e0)
                       error('''eps_opt'' must be a non-negative real number');
                    end
                case 'max_it'
                    max_it = options.max_it;
                    if (~isnumeric(max_it) || ~isreal(max_it) || ~isscalar(max_it) || max_it<1e0)
                       error('''max_it'' must be a real number >= 1');
                    end
                    max_it = floor(max_it);
                case 'verbosity'
                    verbosity = options.verbosity;
                    if (~islogical(verbosity) || ~isscalar(verbosity))
                       error('''verbosity'' must be a logical');
                    end
                otherwise
                    error(['in the fifth input (which is optional) ''' char(opts_field(i)) ''' is not a valid field name']);
            end
        end
    end
    
    r = A*x;
    v = exp(b.*r);
    x_sq = x.*x;
    f = sum(log((1e0./v)+1e0))/l + lambda*sum(x_sq(1:end-1));
    
    gamma = 1e-2;
    delta = 5e-1;
    
    it = 0;
    
    if (max_it < Inf)
        f_vec = zeros(max_it+1,1);
    else
        f_vec = zeros(min(10000,100*n)+1,1);
    end
    f_vec(1) = f;
    g_vec = zeros(length(f_vec)-1,1);
    
    if (verbosity)
        fprintf('%s%.4e\n','it = 0, f = ',f);
    end
	
    while (true)
        
        % compute the gradient
        g = -(A'*(b./(v+1e0)))/l + 2e0*lambda*([x(1:end-1); 0e0]);
        [g_sup_norm,imax] = max(abs(g));
        g_vec(it+1) = norm(g); % not used, just for output
        
        if (g_sup_norm > eps)
            
            if (block_dim > 1)
                ind_block = [0 randperm(n-1,block_dim-1)];
                ind_i = find(ind_block>=imax);
                ind_block(ind_i) = ind_block(ind_i) + 1;
                ind_block(1) = imax;
            else
                ind_block = imax;
            end
            
            x_block = x(ind_block);
            g_block = g(ind_block);
            
            compute_next_point();
            
        else
            
            flag = 0;
            break;
            
        end
        
        it = it + 1;
        f_vec(it+1) = f;
        if (verbosity)
            fprintf('%s%i%s%.4e%s%.4e\n','it = ',it,', f = ',f,', sup norm of g = ',g_sup_norm);
        end
        if (it >= max_it)
            flag = 1;
            break;
        end
        
    end
        
    if (it+1 < max_it)
        f_vec(it+2:end) = [];
        g_vec(it+1:end) = [];
    end
    
    function compute_next_point()
        
        % compute the search direction
        A_block = A(:,ind_block);
        H0 = A_block.*(v./((v+1e0).*(v+1e0)))/l;
        H_reg = ones(block_dim,1);
        i_n = find(ind_block==n);
        if (~isempty(i_n))
            H_reg(i_n) = 0e0;
        end
        diag_H_approx_block = min(max(diag((A_block'*H0))+2e0*lambda*H_reg,1e-2),1e9);
        d = -g_block./diag_H_approx_block;
        
        % Armijo line search
        gamma_gd = gamma*(g_block'*d);
        alpha = 1e0;
        x_block_trial = x_block + d;
        x_sq(ind_block) = x_block_trial.*x_block_trial;
        r_trial = r + A(:,ind_block)*d;
        v_trial = exp(b.*r_trial);
        f_trial = sum(log((1e0./v_trial)+1e0))/l + lambda*sum(x_sq(1:end-1));
        while (f_trial > f+alpha*gamma_gd)
            alpha = delta*alpha;
            x_block_trial = x_block + alpha*d;
            x_sq(ind_block) = x_block_trial.*x_block_trial;
            r_trial = r + alpha*A(:,ind_block)*d;
            v_trial = exp(b.*r_trial);
            f_trial = sum(log((1e0./v_trial)+1e0))/l + lambda*sum(x_sq(1:end-1));
            if (alpha < 1e-20)
                x_block_trial = x_block;
                r_trial = r;
                v_trial = v;
                f_trial = f;
                x_sq(ind_block) = x_block_trial.*x_block_trial;
                break;
            end
        end
        x(ind_block) = x_block_trial;
        r = r_trial;
        v = v_trial;
        f = f_trial;
        
    end
    
end