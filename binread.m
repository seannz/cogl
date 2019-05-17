function data = binread(filename, format, width, height, skip)

    format = lower(format);
    fd = fopen(filename);

    if nargin == 4
        skip = 0;
    end

% if nargin == 4
%     data = reshape(fread(fd, width*height, format),width,height);
% else
    if format(1) == '*'
        format2 = format(2:end);
    else
        format2 = format;
    end

    if strcmp('uint8', format2) || strcmp('int8', format2)
        sz = 1;
    elseif strcmp('uint16', format2) || strcmp('int16', format2)
        sz = 2;
    elseif strcmp('uint32', format2) || strcmp('int32', format2) || ...
            strcmp('float', format2)
        sz = 4;
    elseif strcmp('double', format2)
        sz = 8;
    end

    if fseek(fd, skip*sz, -1) == -1
        error('Seek exceeded file size');
    end
    [data, cnt] = fread(fd, width*height, format);
    if cnt < width*height
        error(['File ' filename ' not large enough']);
    end
    data = reshape(data,width,height)';
% end

fclose(fd);
end