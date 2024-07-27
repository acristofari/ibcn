% This is the main file to solve sparse least-squares problem, as described in the paper
% "A. Cristofari. Block cubic Newton with greedy selection. arXiv:2407.18150"
%
% Note that results might not coincide with those reported in the paper exactly
% due to different software, operating systems, seed initialization, etc.
% 
% Author: Andrea Cristofari (andrea.cristofari@uniroma2.it)
% Last update of this file: July 27th, 2024

% -------------------------------------------------------------------------

% number of problems
n_p = 10;

% problem parameters
lambda = 1e-3;
omega = 1e-2;
p = 5e-1;

% size of blocks
% block_dim = 1;
% block_dim = 5;
% block_dim = 10;
% block_dim = 20;
% block_dim = 50;
block_dim = 100;
    
% -------------------------------------------------------------------------

% initialize to store results
x_ibcn = cell(n_p,1);
f_ibcn = cell(n_p,1);
g_ibcn = cell(n_p,1);
t_ibcn = cell(n_p,1);
it_ibcn = zeros(n_p,1);
flag_ibcn = zeros(n_p,1);

eps = 0e0;
max_it = 10000;

opts_ibcn = struct('block_dim',block_dim,'eps_opt',eps,'max_it',max_it);
opts_bcd = opts_ibcn;

rng(block_dim)

for i_p = 1:n_p

    % create the problem
    n = 10000;
    l = 500;
    spc = 0.05;
    fprintf('%s%i%s%i%s%.4e%s\n','Creating instance (l = ', l, ', n = ', n, ', spc = ', spc, ')...')
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
    [x_ibcn{i_p},it_ibcn(i_p),flag_ibcn(i_p),f_ibcn{i_p},g_ibcn{i_p}] = ibcn_sp_ls(A,b,x0,lambda,omega,p,opts_ibcn);
    fprintf(['***********************************************************' ...
             '\nAlgorithm: INEXACT BLOCK CUBIC NEWTON' ...
             '\nf =  %-.4e'   ...
             '\n||g|| = %-.4e'   ...
             '\niterations = %-i' ...
             '\nflag = %-i' ...
             '\n***********************************************************\n\n'], ...
            f_ibcn{i_p}(end),g_ibcn{i_p}(end),it_ibcn(i_p),flag_ibcn(i_p));
    %----------------------------------------------------------------------
        
end