clearvars;
close all;

rng(0);
bandbeta = 'results_dim_10_old.mat';

% % sequence
fileorig = 'stockholm_640x360_59.9401.yuv';
dims=[360,640];

% % tunable parameters
run = 200; % number of iterations
sig = 7.0000;
rho = 2.0000;
tau = 38.000;
alp = 140.00;
qlt = 10.000; % quantization factor
dim = [10,0,0];
frames = 3;
frskip = 0;
% % end tunatble parameters

%optimization
tic;
graphmpeg_fun(bandbeta,dim,fileorig,dims,qlt,alp,sig,tau,rho,frames,frskip,run,true)
toc;
