function results = graphmpeg_fun(bandbeta, dim, fileorig, dims, qlt, alp, sig, tau, rho, frames, frskip, run, show)
    if nargin < 9
        show = false;
    end

    tgt_idx = (frames + 1) / 2;

    filtsigs = 'sigmas.mat';
    load(bandbeta,'final_result','sigs','taus');
    load(filtsigs,'sigmas');

    %% tunable parameters
    % moved to _scan function
    %dim = [dim,0,0];

    % Inverse Lipschitz constant of RtLR
    beta = 1./squeeze(final_result(find(taus==tau,1),find(sigs==sig,1),:)).^2;
    L = beta'*reshape(sigmas(:,2)*sigmas(:,2)',[],1);
    t = 1/L;

    n = 8;
    J = compress(['./vid/' fileorig],qlt,frames,frskip,dims);

    beta = t*beta;
    xiQ2x = (12*t*alp)./(J.quant_tables{1}.^2);

    v_orig = yuv_comp_read(['./vid/' fileorig],'uint8',dims(2),dims(1),frames,frskip)-128;
    yy = decompress(J,1);

    u     = cell(2,1);
    u{1} = yy + 1e-100; %force memcpy
    u{2} = yy + 1e-100; %force memcpy
    z     = zeros(size(yy),'single');
    Hx    = zeros([n^2,size(yy)],'single');
    HtLHx = zeros(size(yy),'single');

    % initialize x
    x = yy + 1e-100;

    % set up position vectors
    zig = zig8x8;
    psi = sqrt(J.quant_tables{1}(zig(1:dim(1))).^2./12 + tau^2);

    fct2(yy,Hx);

    x_hat = zeros(size(yy),'single');
    fct8x8(yy,x_hat);
    ub = x_hat + repmat(0.5*J.quant_tables{1},size(x_hat)./[8,8,1]);
    lb = x_hat - repmat(0.5*J.quant_tables{1},size(x_hat)./[8,8,1]);

    xiQ2x_hat = blockproc(x_hat,[8,8],@(block) block.data .* xiQ2x);
    IxiQ2_inv = repmat(1./(1 + xiQ2x),size(x_hat)./[8,8,1]);

    ref = bsxfun(@times,Hx(zig(1:dim),:,:,:),1.0./psi);

    % symmetric normalizer
    Hx(:) = 1;
    LHx = nltemps(Hx,ref,sig,rho,1,0);
    D_dag = 1./LHx(1,:,:,:);

    results = zeros(run+1,2);
    results(1,1) = psnr(min(max(yy(:,:,tgt_idx),-128),127),v_orig(:,:,tgt_idx),255);
    results(1,2) = ssim(min(max(yy(:,:,tgt_idx)+128,0),255),v_orig(:,:,tgt_idx)+128,'DynamicRange',255);
    if show
        fprintf('%3d %6.4f %1.4f\n', 1, results(1,1), results(1,2));
    end
    
    img_name = sprintf(['./vid_recon/' fileorig '_%d_%d_%d_'], ...
        frskip + tgt_idx - 1, frames, qlt);
    imwrite(yy(:,:,tgt_idx) / 255 + 0.5, [img_name 'start.png']);

    for k=1:run
        %-% projected gradient descent
        p = mod(k,2);

        %-% descent step: see eq. (26) and  Fig. 4.
        %  operator H
        fct2(x,Hx);
        % operator D^dag L
        AHx = nltemps(Hx,ref,sig,rho,1,1);
        LHx = Hx - D_dag .* AHx;
        % operator L D^dag
        ADLHx = nltemps(D_dag .* LHx,ref,sig,rho,1,1);
        % multiply lambda*(beta_1 ... beta_64)
        LHx = beta .* (LHx - ADLHx);
        % operator H^transpose
        ifct2(LHx,HtLHx);
        % descend, see eq. (26)
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

        results(k+1, 1) = psnr(min(max(x(:,:,tgt_idx),-128),127),v_orig(:,:,tgt_idx),255);
        results(k+1, 2) = ssim(min(max(x(:,:,tgt_idx)+128,0),255),v_orig(:,:,tgt_idx)+128,'DynamicRange',255);
        if show
            fprintf('%3d %6.4f %1.4f\n', k+1, results(k+1,1), results(k+1,2));
        end        
    end
    imwrite(x(:,:,tgt_idx)/255 + 0.5, [img_name 'end.png']);
end
