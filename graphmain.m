% clear all;
% close all;

%addpath('~/Developer/gkdtrees/matlab');

bandbeta = 'results_dim_10_old.mat';
%file = 'kodim20.png';            % ground truth for psnr
file = 'cameraman.png';

%% tunable parameters
run = 200; 
sig = 7.0000;
tau = 38.000;
alp = 128.0000;
dim = 10;
qlt = 10.000; % quantization factor
% end tunatble parameters

%% prepare the file outside the optimization loop
% this way a gray-scale image is processed inside
dot = strfind(file, '.');        % find where the . is
filecomp = sprintf('scratch/%s_%02d.jpg', file(1:dot(end)-1), qlt);
fileorig = sprintf('img/%s', file); % ground truth for psnr
src = im2double(rgb2gray(imread(fileorig)));
imwrite(src, filecomp, 'jpg', 'Quality', qlt);
% end of preparation

%% optimization
tic;
graphmain_fun3(bandbeta, dim, file, qlt, alp, sig, tau, run, true);
toc;
