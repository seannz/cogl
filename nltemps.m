function [out] = nltemps(img,gud,sig_s,sig_t,sig_c,replay)
    if nargin == 4
        replay = 0;
    end
    img = single(permute(img,[1,2,3,4]));
    gud = single(permute(gud,[1,2,3,4]));
    out = nltemps_mex(img,gud,sig_s,sig_t,sig_c,replay);
    out = single(permute(out,[1,2,3,4]));
end