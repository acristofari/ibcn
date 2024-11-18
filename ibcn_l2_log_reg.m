% Inexact Block Cubic Newton (IBCN) method with greedy selection for l2-regularized logistic regression (labels -1 and 1), as described in the paper
% "A. Cristofari. Block cubic Newton with greedy selection. arXiv:2407.18150".
%
% Author: Andrea Cristofari (andrea.cristofari@uniroma2.it)
% Last update of this file: November 18th, 2024
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

function [f,x,it,flag,f_vec,g_vec,t_vec] = ibcn_l2_log_reg(A,b,x,lambda,options)

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
    
    % set options
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
    
    eta = 1e-1;
    gamma = 2;
    tau = 1e0;
    
    r = A*x;
    v = exp(b.*r);
    x_sq = x.*x;
    f = sum(log((1e0./v)+1e0))/l + lambda*sum(x_sq(1:end-1));
            
    s = 0e0;
    s_norm = 0e0;
    M_block = 1e0;
    H_block = 0e0;
    flag_cubic = 0;
    ms = 0e0;
    g_m = 0e0;
    g_m_norm = 0e0;
    alpha = 0e0;
    
    it = 0;
    
    if (max_it < Inf)
        t_vec = zeros(max_it+1,1);
    else
        t_vec = zeros(min(10000,100*n)+1,1);
    end
    f_vec = [f; -Inf(length(t_vec)-1,1)];
    g_vec = -Inf(length(t_vec)-1,1);
    
    t0 = tic;
	
    if (verbosity)
        fprintf('%s%.4e\n','it = 0, f = ',f);
    end
	
    while (true)
        
        % compute the gradient       
        g = -(A'*(b./(v+1e0)))/l + 2e0*lambda*([x(1:end-1); 0e0]);
        [g_sup_norm,imax] = max(abs(g));
                
        g_norm = norm(g);
        g_vec(it+1) = g_norm;
        
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
            
            it_succ = compute_next_point;
            if ~it_succ
                M_block = gamma*M_block;
            end
                        
        else
            
            flag = 0;
            break;
            
        end
        
        it = it + 1;
        t = toc(t0);
        t_vec(it+1) = t;
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
        t_vec(it+2:end) = [];
    end
    
    function it_succ = compute_next_point()
        
        A_block = A(:,ind_block);
        H0 = A_block.*(v./((v+1e0).*(v+1e0)))/l;
        H_reg = eye(block_dim);
        i_n = find(ind_block==n);
        if (~isempty(i_n))
            H_reg(i_n,i_n) = 0e0;
        end
        H_block = (A_block'*H0) + (2e0*lambda)*H_reg;
        
        it_succ = false;
        x_block_old = x_block;
        x_sq_block_old = x_sq(ind_block);
        r_old = r;
        v_old = v;
        f_old = f;
         
        % minimize the cubic model (approximately) starting from the Cauchy point
        
        flag_cubic = 0;
        
        sq_norm_g_block = g_block'*g_block;
        gHg_block = g_block'*H_block*g_block;
        alpha = 2e0*sq_norm_g_block/(gHg_block+sqrt(gHg_block*gHg_block+2e0*M_block*sq_norm_g_block*sq_norm_g_block*sqrt(sq_norm_g_block)));
        s = -alpha*g_block;
        
        s_norm = sqrt(s'*s);
        Hs_block = H_block*s;
        M_snorm_s = M_block*s_norm*s;
        g_m = g_block + Hs_block + 5e-1*M_snorm_s;
        g_m_norm = sqrt(g_m'*g_m);
        
        if (g_m_norm>tau*s_norm*s_norm)
            ms = g_block'*s + 5e-1*(Hs_block'*s) + (M_snorm_s'*s)/6e0;
            [qs,flag_cubic] = minimize_cubic_model(Hs_block);
        else
            qs = s'*(g_block+5e-1*Hs_block);
        end
        if (flag_cubic < 5e-1)
            x_block = x_block_old + s;
            r = r + A(:,ind_block)*s;
            v = exp(b.*r);
            x_sq(ind_block) = x_block.*x_block;
            f = sum(log((1e0./v)+1e0))/l + lambda*sum(x_sq(1:end-1));
            if ((f-f_old<=eta*qs) && (qs<0e0))
                it_succ = true;
                x(ind_block) = x_block;
            else
                x_sq(ind_block) = x_sq_block_old;
                r = r_old;
                v = v_old;
                f = f_old;
            end
        end
                
    end
    
    function [qs,flag_cubic] = minimize_cubic_model(Hs)
        
        %--------------------------%
        % SPECTRAL GRADIENT METHOD %
        %--------------------------%
        
        max_it_cubic = 10000;
        
        % direction parameters
        c_bb_min = 1e-10;
        c_bb_max = 1e10;
        
        % line search parameters
        gamma_ls = 1e-2;
        delta_ls = 5e-1;
        ls_mem = 1;
        mw = ms;
        w = mw*ones(ls_mem,1);
        
        % initialize parameters for spectral gradient
        d_m = -g_block;
        g_m_old = g_block;
                
        it_inner = 1;
        flag_cubic = 0;
        
        while (true)
            
            % compute direction
            y = g_m - g_m_old;
            dy = d_m'*y;
            if (dy > 0e0)
                c_bb = alpha*(d_m'*d_m)/dy;
                if (c_bb < c_bb_min)
                    c_bb = alpha*dy/(y'*y);
                    if (c_bb < c_bb_min)
                        c_bb = c_bb_min;
                    else
                        c_bb = min(c_bb,c_bb_max);
                    end
                else
                    c_bb = min(c_bb,c_bb_max);
                end
            else
                c_bb = min(c_bb_max,max(1e0,s_norm/g_m_norm));
            end
            d_m = -c_bb*g_m;
                     
            % Armijo line search
            alpha = 1e0;
            s_trial = s + d_m;
            s_trial_norm = sqrt(s_trial'*s_trial);
            Hd = H_block*d_m;
            Hs_trial = Hs + Hd;
            M_snorm_s_trial = M_block*s_trial_norm*s_trial;
            qs_trial = s_trial'*g_block + 5e-1*(s_trial'*Hs_trial);
            ms_trial = qs_trial + (s_trial'*M_snorm_s_trial)/6e0;
            gamma_gd_m = gamma_ls*(g_m'*d_m);
            while (ms_trial > mw + alpha*gamma_gd_m)
                alpha = delta_ls*alpha;
                if (alpha < 1e-20)
                    flag_cubic = 1;
                end
                s_trial = s + alpha*d_m;
                s_trial_norm = sqrt(s_trial'*s_trial);
                Hs_trial = Hs + alpha*Hd;
                M_snorm_s_trial = M_block*s_trial_norm*s_trial;
                qs_trial = s_trial'*g_block + 5e-1*(s_trial'*Hs_trial);
                ms_trial = qs_trial + (s_trial'*M_snorm_s_trial)/6e0;
            end
            
            if (flag_cubic > 0)
                s = zeros(size(s));
                qs = 0e0;
                break;
            end
            
            % set the new point
            g_m_old = g_m;
            s = s_trial;
            ms = ms_trial;
            qs = qs_trial;
            Hs = Hs_trial;
            g_m = g_block + Hs + 5e-1*M_snorm_s_trial;
            s_norm = s_trial_norm;
            g_m_norm = sqrt(g_m'*g_m);
            w = [ms; w(1:end-1)];
            mw = max(w);
            
            if (g_m_norm <= tau*s_norm*s_norm)
                break;
            end
            
            if (s_norm < 1e-12)
                flag_cubic = 1;
                s = zeros(size(s));
                qs = 0e0;
                break;
            end
            
            it_inner = it_inner + 1;
            
            if (it_inner > max_it_cubic)
                flag_cubic = 1;
                s = zeros(size(s));
                qs = 0e0;
                break;
            end
        end
        
    end
    
end