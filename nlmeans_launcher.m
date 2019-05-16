close all;
clear all;

filename = 'korea.jpg';

matlab_dir = [pwd];
source_dir = [pwd '/../../tip2018_bilateral/source/'];
output_dir = [pwd '/../../tip2018_bilateral/var/'];

sig_S = [4,6,8,12,16,24,32,48,64,96,128];
sig_C = 96;

pca = [6];
img = single(imread([source_dir filename]));
img(:,:,4) = 1;

snrs = zeros(2,length(pca),length(sig_S),length(sig_C));
secs = zeros(4,length(pca),length(sig_S),length(sig_C));

lensigc = length(sig_C);
lensigs = length(sig_S);
lenpcas = length(pca);
for p = 1:lenpcas
    gud = compute_non_local_means_basis(img(:,:,1:3),4,pca(p));
    for s = lensigs:-1:1
        for c = lensigc:-1:1
            sig_s = sig_S(s);
            sig_c = sig_C(c);
            load([output_dir filename '_' num2str(s) '_' ...
                  num2str(c) '_nl'],'-mat');
            [res,sec] = nlmeans(img,gud,sig_s,sig_c);
            [psr,snr] = psnr(res(:,:,1:3).*(1./res(:,:,4)),out(:,:,1:3)./out(:,:,4),255);
            snrs(:,p,s,c) = [psr,snr];
            secs(:,p,s,c) = sec;
        end
    end
end
save('nlmeans_results.mat', 'pca', 'snrs', 'secs');
