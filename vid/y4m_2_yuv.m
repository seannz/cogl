% converts y4m file to yuv
% a large percentage of this file was copied from yuv4mpeg2mov.m
function [mov, fields, accepted] = y4m_2_yuv(File, tgt_fname_pre)
    filep = dir(File); 
    fileBytes = filep.bytes; %Filesize
    clear filep;
    
	inFileId = fopen(File, 'r');
	[header, endOfHeaderPos] = textscan(inFileId,'%s',1,'delimiter','\n');
    [fields, accepted] = readYuv4MpegHeader(header);
    fields.frameCount = 0;
    
	frameLength = fields.width * fields.height;    
    if strcmp(fields.colourSpace(1:4), 'C420')
		frameLength = (frameLength * 3) / 2;
	elseif strcmp(fields.colourSpace, 'C422')
		frameLength = (frameLength * 2);
	elseif strcmp(fields.colourSpace, 'C444')
		frameLength = (frameLength * 3);
    end

    % compute number of frames, a frame starts with FRAME and then some
    % possible parameters and finally ending with the byte 0x0A.
    % Assume no parameters
    frameCount = (fileBytes - endOfHeaderPos)/(6 + frameLength);
    
    if mod(frameCount,1) ~= 0
        disp('Error: wrong resolution, format or filesize');
        accepted = false;
    else
        fields.frameCount = frameCount;
        
        %synthesize target file name
        t = find(File == '_');
        tgt_fname = [tgt_fname_pre '_' ...
            num2str(fields.width) 'x' num2str(fields.height) '_' ...
            num2str(fields.fps) '.yuv'];
        outFileId = fopen(tgt_fname, 'w');
        for framenumber = 1:frameCount
            fread(inFileId, 6, 'uchar');
            yuv = fread(inFileId, frameLength, 'uchar');
            fwrite(outFileId, yuv);
        end
        fclose(outFileId);
    end
    
	fclose(inFileId);
end

function [fields, accepted] = readYuv4MpegHeader(header)
	colourSpace = 'C420';

	parts = strsplit(char(header{1}), ' ');

    accepted = strcmp(parts{1}, 'YUV4MPEG2');
	assert(accepted, 'file must start with YUV4MPEG2');

	width = textscan(parts{2}, 'W%n');
	height = textscan(parts{3}, 'H%n');

	fpsFraction = textscan(parts{4}, 'F%n:%n');
	fps = fpsFraction{1} / fpsFraction{2};

	interlacing = textscan(parts{5}, 'I%c');

	pixelAspectFraction = textscan(parts{6}, 'A%n:%n');
    pixelAspectRatio = pixelAspectFraction{1} / pixelAspectFraction{2};
    
    if size(parts,2) > 6 && strfind(parts{7}, 'C')
        colourSpace = parts{7};
    end
    
    fields = struct('width', width,...
        'height', height,...
        'fps', fps,...
        'interlacing',interlacing,...
        'pixelAspectRatio',pixelAspectRatio,...
        'colourSpace',colourSpace);
end