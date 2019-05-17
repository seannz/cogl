function vid = yuv_comp_read(filename, filetype, width, height,frames, skip)
    if nargin < 6
        skip = 0;
    end

    vid = zeros(height,width,frames,'single');
    
    for k = 1:frames
        % assume 420 yuv
        s = 1.5*(k-1+skip) * width * height;
        vid(:,:,k) = single(binread(filename,filetype,width,height,s));
    end
end