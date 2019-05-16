function [results, y_recon] = graphmain_fun3(bandbeta, dim, file, qlt, alp, sig, tau, run, show)
    if nargin < 9
        show = false;
    end
                                         
    % preparation
    dot = strfind(file, '.');        % find where the . is
    assert(isempty(dot) == false);   % must have an extension
    
    rng(0);
    dctsigs  = 'dctsigs.mat';
    filtsigs = 'sigmas.mat';
    filecomp = sprintf('scratch/%s_%02d.jpg', file(1:dot(end)-1), qlt);
    fileorig = sprintf('img/%s', file); % ground truth for psnr

    load(bandbeta,'final_result','sigs','taus');
    load(filtsigs);
    load(dctsigs,'vars');
    n = 8;
    % color transform before compression (gray-scale image)
%     src = im2double(rgb2gray(imread(fileorig))); 
%     imwrite(src, filecomp, 'jpg', 'Quality', qlt);
    % no color transform (color image)
%     imwrite(imread(fileorig),filecomp,'Quality',qlt);
    beta = 1./squeeze(final_result(find(taus==tau,1),find(sigs==sig,1),:)).^2;

    % ground truth
    v_orig = single(rgb2ycc(double(imread(fileorig))));
    v_orig = v_orig(:,:,1) - 128;

    % decode coeff
    J = jpeg_read(filecomp);
    % decode image
    yy = single(dct8x8(J.coef_arrays{1},J.quant_tables{1}));

    % Lipschitz constant of HtLDDLH, see eq. (35)
    L = beta'*reshape(sigmas(:,2)*sigmas(:,2)',[],1);
    % optimal descent step size, see eq. (32)
    lam = 1/L;

    beta = lam*beta;

    u     = cell(2,1);
    u{1} = yy + 1e-100;
    u{2} = yy + 1e-100;
    Hx    = zeros([n^2,size(yy)],'single');
    z     = zeros(size(yy),'single');
    HtLHx = zeros(size(yy),'single');

    % initialize x
    x = yy + 1e-100;

    % set up position vectors
    zig = zig8x8;
    psi = sqrt(J.quant_tables{1}(zig(1:dim)).^2./12 + tau^2);

    fct2(yy,Hx);

    x_hat = zeros(size(yy),'single');
    % bin centers, see eq. (3)
    fct8x8(yy,x_hat);
    % bin boundaries, see eq. (4)
    ub = x_hat + repmat(0.5*J.quant_tables{1},size(x_hat)/8);
    lb = x_hat - repmat(0.5*J.quant_tables{1},size(x_hat)/8);

    xiQ2 = (12*lam*alp)./(J.quant_tables{1}.^2);
    xiQ2x_hat = blockproc(x_hat,[8,8],@(block) block.data .* xiQ2);

    IxiQ2_inv = repmat(1./(1 + xiQ2),size(xiQ2x_hat)/8);

    ref = bsxfun(@times,Hx(zig(1:dim),:,:),1.0./psi);

    % symmetric normalizer
    Hx(:) = 1;
    LHx = nlmeans(Hx,ref,sig,1,0);
    D_dag = 1./LHx(1,:,:);

    results = zeros(1, run+1, 2);
    results(1, 1, :) = [psnr(min(max(yy,-128),127),v_orig,255) ...
        ssim(min(max(yy+128,0),255),v_orig+128,'DynamicRange',255)];
    if show
        fprintf('%3d %6.4f %1.4f\n', 0, results(1,1,1), results(1,1,2));
    end
    
    for k=1:run
        %-% projected gradient descent
        p = mod(k,2);

        %-% descent step: see eq. (26) and  Fig. 4.
        %  operator H
        fct2(x,Hx);
        %  operator D^dag L
        AHx = nlmeans(Hx,ref,sig,1,1);
        LHx = Hx - D_dag .* AHx;
        %  operator L D^dag 
        ADLHx = nlmeans(D_dag .* LHx,ref,sig,1,1);
        %  multiply lambda*(beta_1 ... beta_64)
        LHx = beta .* (LHx - ADLHx);
        %  operator H^transpose
        ifct2(LHx,HtLHx);
        %  descend, see eq. (26)
        y = x - HtLHx;

        %-% projection step: see eq. (27) and Fig. 5
        %  z = Ty, see just above eq. (29)
        fct8x8(max(-128,min(127,y)),z);
        %  u = T^*Proj[h(z)], see eq. (31) and the note above eq. (29)
        ifct8x8(max(min(IxiQ2_inv.*(z+xiQ2x_hat),ub),lb),u{1+p});

        %-% extrapolation step: see eq. (25)
        %  see eq. (32)
        theta = k/(k+3);
        x = u{1+p} + theta*(u{1+p}-u{2-p});
        
        results(1, k+1, :) = [psnr(min(max(x,-128),127),v_orig,255) ...
                              ssim(min(max(x+128,0),255),v_orig+128,'DynamicRange',255)];

        if show
            fprintf('%3d %6.4f %1.4f\n', k, results(1,k+1,1), results(1,k+1,2));
        end
    end
    y_recon = double(min(max(x+128,0),255) / 255);
end