function g = dct8x8(u,q)
    [m,n,p] = size(u);
    g = zeros(m,n,p,'single');
    for i = 1:p
        g(:,:,i) = blockproc(u(:,:,i),[8,8],@(block) idct2(block.data.*q,[8,8]));
    end
    %g = g + 128;
    %g = max(0,min(255,g + 128));
    %g = max(-128, min(127, g)) + 128;
end