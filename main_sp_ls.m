% This is the main file to solve sparse least squares problem, as described in the paper
% "A. Cristofari. Block cubic Newton with greedy selection. arXiv:2407.18150"
%
% Note that results may not coincide with those reported in the paper exactly
% due to different software, operating systems, seed initialization, etc.
% 
% Author: Andrea Cristofari (andrea.cristofari@uniroma2.it)
% Last update of this file: October 13th, 2025

% -------------------------------------------------------------------------

% number of problems
n_p = 2;

% problem parameters
lambda = 1e-2;
omega = 1e-2;
p = 5e-1;

% size of blocks
block_dim = 1;
% block_dim = 5;
% block_dim = 10;
% block_dim = 20;
% block_dim = 50;
% block_dim = 100;
    
% -------------------------------------------------------------------------

% initialization to store results
x_ibcn = cell(n_p,1);
f_ibcn = zeros(n_p,1);
g_ibcn = cell(n_p,1);
it_ibcn = zeros(n_p,1);
flag_ibcn = zeros(n_p,1);

% algorithm parameters
eps = 1e-3; % optimality precision
max_it = 10000; % maximum number of iterations
store_matrix = true; % to store A^T A, it may be impractical if A is too large

% create options for ibcn
opts_ibcn = struct('block_dim',block_dim,'eps_opt',eps,'max_it',max_it,'store_matrix',store_matrix);

for i_p = 1:n_p
    
    % create the problem
    n = 1000;
    l = n;
    spc = 0.01;
    fprintf('%s%i%s%i%s%.4e%s\n','Creating instance (l = ', l, ', n = ', n, ', spc = ', spc, ')...')
    rng(100+i_p);
    A = rand(l,n);
    q = randperm(n);
    xstar = zeros(n,1);
    T = floor(spc*l);
    xstar(q(1:T)) = sign(randn(T,1));
    sigma = 0.001;
    b = A*xstar + sigma*randn(l,1);
    fprintf('%s\n\n','Done.')
    
    x0 = zeros(n,1); % starting point
    
    %----------------------------------------------------------------------
    % INEXACT BLOCK CUBIC NEWTON
    %----------------------------------------------------------------------
    [f_ibcn(i_p),x_ibcn{i_p},it_ibcn(i_p),t_ibcn(i_p),flag_ibcn(i_p)] = ibcn_sp_ls(A,b,lambda,omega,p,opts_ibcn);
    fprintf(['***********************************************************' ...
             '\nAlgorithm: INEXACT BLOCK CUBIC NEWTON' ...
             '\nf = %-.4e'   ...
             '\niterations = %-i' ...
             '\nflag = %-i' ...
             '\n***********************************************************\n\n'], ...
            f_ibcn(i_p),it_ibcn(i_p),flag_ibcn(i_p));
    %----------------------------------------------------------------------
    
end