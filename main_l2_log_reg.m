% This is the main file to solve l2-regularized logistic regression, as described in the paper
% "A. Cristofari. Block cubic Newton with greedy selection. arXiv:2407.18150"
%
% Note that results might not coincide with those reported in the paper exactly
% due to different software, operating systems, seed initialization, etc.
% 
% Author: Andrea Cristofari (andrea.cristofari@uniroma2.it)
% Last update: July 26th, 2024

% -------------------------------------------------------------------------

% problems to be previously downloaded from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
% and converted into matlab files, named as follows, by using the libsvmread software,
% which can be downloaded from therein as well, such that the sparse matrix A is the instance matrix and b is the label vector

problem_set = {
               'gisette'  % not to be scaled
               'leu';     % not to be scaled
               'madelon'; % to be scaled
              };
n_p = length(problem_set);
scale = [false; false; true];

% problem parameters
lambda = 1e-3;
%--------------------------------------------------------------------------

% size of blocks
% block_dim = 1;
% block_dim = 5;
% block_dim = 10;
% block_dim = 20;
% block_dim = 50;
block_dim = 100;

% initialize to store results
x_ibcn = cell(n_p,1);
f_ibcn = cell(n_p,1);
g_ibcn = cell(n_p,1);
t_ibcn = cell(n_p,1);
it_ibcn = zeros(n_p,1);
flag_ibcn = zeros(n_p,1);
x_bcd = cell(n_p,1);
f_bcd = cell(n_p,1);
g_bcd = cell(n_p,1);
t_bcd = cell(n_p,1);
it_bcd = zeros(n_p,1);
flag_bcd = zeros(n_p,1);

eps = 0e0;
max_it = 10000;

opts_ibcn = struct('block_dim',block_dim,'eps_opt',eps,'max_it',max_it);
opts_bcd = opts_ibcn;

rng(block_dim)
    
for i_p = 1:n_p
    
    dataset_name = problem_set{i_p};
    fprintf('%s%s%s\n','Loading ', dataset_name, '...');
    
    load([dataset_name '.mat'],'A','b');
    A = full(A);
    
    [m,n] = size(A); 
    fprintf('%s%i%s%i%s\n\n','Done (number of samples = ', m, ', number of features = ', n, ')')
    if (scale(i_p)) % scale in [0,1]
        feat_min = full(min(A));
        feat_max = full(max(A));
        ind_feat_scale = (feat_min<feat_max);
        feat_min = feat_min(ind_feat_scale);
        feat_max = feat_max(ind_feat_scale);
        A(:,ind_feat_scale) = (A(:,ind_feat_scale)-repmat(feat_min,size(A,1),1))./(repmat(feat_max-feat_min,size(A,1),1));
    end
    
    % check if labels are -1 and +1
    if (length(unique(b)) ~= 2)
        error('only binary classification is allowed');
    end
    if (any(abs(b)~=1))
        idx = find(b==min(b));
        b(idx) = -1;
        b(~idx) = 1;
    end
    
    x0 = zeros(n+1,1);
    
    %----------------------------------------------------------------------
    % INEXACT BLOCK CUBIC NEWTON
    %----------------------------------------------------------------------
    [x_ibcn{i_p},it_ibcn(i_p),flag_ibcn(i_p),f_ibcn{i_p},g_ibcn{i_p}] = ibcn_l2_log_reg(A,b,x0,lambda,opts_ibcn);
    fprintf(['***********************************************************' ...
             '\nAlgorithm: INEXACT BLOCK CUBIC NEWTON' ...
             '\nf =  %-.4e'   ...
             '\n||g|| = %-.4e'   ...
             '\niterations = %-i' ...
             '\nflag = %-i' ...
             '\n***********************************************************\n\n'], ...
            f_ibcn{i_p}(end),g_ibcn{i_p}(end),it_ibcn(i_p),flag_ibcn(i_p));
    %----------------------------------------------------------------------
    
            
    %----------------------------------------------------------------------
    % 2nd ORDER BLOCK COORDINATE DESCENT
    %----------------------------------------------------------------------
    [x_bcd{i_p},it_bcd(i_p),flag_bcd(i_p),f_bcd{i_p},g_bcd{i_p}] = bcd_l2_log_reg(A,b,x0,lambda,opts_bcd);
    fprintf(['***********************************************************' ...
             '\nAlgorithm: 2nd ORDER BLOCK COORDINATE DESCENT' ...
             '\nf =  %-.4e'   ...
             '\n||g|| = %-.4e'   ...
             '\niterations = %-i' ...
             '\nflag = %-i' ...
             '\n***********************************************************\n\n'], ...
            f_bcd{i_p}(end),g_bcd{i_p}(end),it_bcd(i_p),flag_bcd(i_p));
    %----------------------------------------------------------------------
         
end