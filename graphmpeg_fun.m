function results = graphmpeg_fun(bandbeta, dim, fileorig, dims, qlt, alp, sig, tau, rho, frames, frskip, run, show)
    if nargin < 9
        show = false;
    end

    tgt_idx = (frames + 1) / 2;

    filtsigs = 'sigmas.mat';
    load(bandbeta,'final_result','sigs','taus');
    load(filtsigs,'sigmas');

    %% tunable parameters
    accuracy = 1.0; %default is 0.5
    rng(0);
    rnd = rand(300000,1,'single');

    % Inverse Lipschitz constant of RtLR
    beta = 1./squeeze(final_result(find(taus==tau,1),find(sigs==sig,1),:)).^2;
    L = 8*beta'*reshape(sigmas(:,2)*sigmas(:,2)',[],1);
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
    Ry    = zeros([n^2,size(yy)],'single');
    Ry2   = zeros([n^2,size(yy)],'single');
    LRy   = zeros([n^2,size(yy)],'single');
    LRy2  = zeros([n^2,size(yy)],'single');
    RtLRy = zeros(size(yy),'single');
    
    % initialize x
    u{1} = yy + 1e-100; %force memcpy
    u{2} = yy + 1e-100; %force memcpy
    y = yy + 1e-100;

    % set up position vectors
    zig = zig8x8;
    psi = sqrt(J.quant_tables{1}(zig(1:dim(1))).^2./12 + tau^2);

    splat =  4+accuracy*64;
    slice = 12+accuracy*64;
    smear = 16+accuracy*128;
    
    fct2(yy,Ry);

    x_hat = zeros(size(yy),'single');
    fct8x8(yy,x_hat);
    ub = x_hat + repmat(0.5*J.quant_tables{1},size(x_hat)./[8,8,1]);
    lb = x_hat - repmat(0.5*J.quant_tables{1},size(x_hat)./[8,8,1]);

    xiQ2x_hat = blockproc(x_hat,[8,8],@(block) block.data .* xiQ2x);
    IxiQ2_inv = repmat(1./(1 + xiQ2x),size(x_hat)./[8,8,1]);

    deg = zeros([1,size(y)],'single');

    [h,w,d] = size(y);
    ref = zeros([sum(dim)+2,size(y)],'single');
    ref(1,:,:,:) = repmat(permute(1:h,[2,1,3])*(1.0./sig),[1,w,d]);
    ref(2,:,:,:) = repmat(permute(1:w,[1,2,3])*(1.0./sig),[h,1,d]);
    ref(3,:,:,:) = repmat(permute(1:d,[1,3,2])*(1.0./rho),[h,w,1]);
    ref(3+(1:dim(1)),:,:,:) = bsxfun(@times,Ry(zig(1:dim(1)),:,:,:),1.0./psi);

    leaves = fgt2(ref);

    % splat nodes
    splatres = zeros([1,size(yy)],'int32');
    splatids = zeros([splat,size(yy)],'int32');
    splatwgs = zeros([splat,size(yy)],'single');

    % slice nodes
    sliceres = zeros([1,size(yy)],'int32');
    sliceids = zeros([slice,size(yy)],'int32');
    slicewgs = zeros([slice,size(yy)],'single');

    % smear nodes
    smearres = zeros([1,leaves],'int32');
    smearids = zeros([smear,leaves],'int32');
    smearwgs = zeros([smear,leaves],'single');

    % smear data
    oldvals = zeros([64,leaves],'single');
    newvals = zeros([64,leaves],'single');

    % symmetric normalizer
    fgt2(ref,Ry,LRy,deg,rnd,...
         splatids,splatwgs,splatres,...
         sliceids,slicewgs,sliceres,...
         smearids,smearwgs,smearres,oldvals,newvals,accuracy,0);
    sym = 1./deg;

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
        fct2(y,Ry);
        Ry = bsxfun(@times,Ry,beta);
        fgt2(ref,Ry,LRy,deg,rnd,...
             splatids,splatwgs,splatres,...
             sliceids,slicewgs,sliceres,...
             smearids,smearwgs,smearres,...
             oldvals,newvals,accuracy,leaves);
        LRy = Ry - bsxfun(@times,LRy,sym);
        Ry2 = bsxfun(@times,LRy,sym);
        fgt2(ref,Ry2,LRy2,deg,rnd,...
             splatids,splatwgs,splatres,...
             sliceids,slicewgs,sliceres,...
             smearids,smearwgs,smearres,...
             oldvals,newvals,accuracy,leaves);
        LRy = LRy - LRy2;
        % take adjoint DCT
        ifct2(LRy,RtLRy);
        % descend
        y = y - RtLRy;

        %-% projection step: see eq. (27) and Fig. 5
        %  z = Ty, see just above eq. (29)
        fct8x8(max(-128,min(127,y)),y);
        %  u = T^*Proj[h(z)], see eq. (31) and the note above eq. (29)
        ifct8x8(max(min(IxiQ2_inv.*(y+xiQ2x_hat),ub),lb),u{1+p});

        %-% extrapolation step: see eq. (25)
        %  see eq. (32)
        theta = k/(k+3);
        y = u{1+p} + theta*(u{1+p}-u{2-p});

        results(k+1, 1) = psnr(min(max(y(:,:,tgt_idx),-128),127),v_orig(:,:,tgt_idx),255);
        results(k+1, 2) = ssim(min(max(y(:,:,tgt_idx)+128,0),255),v_orig(:,:,tgt_idx)+128,'DynamicRange',255);
        if show
            fprintf('%3d %6.4f %1.4f\n', k+1, results(k+1,1), results(k+1,2));
        end        
    end
    imwrite(y(:,:,tgt_idx)/255 + 0.5, [img_name 'end.png']);
end
