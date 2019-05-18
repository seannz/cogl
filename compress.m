function J = compress(fileorig,quality,frames,frskip,dims)
    if quality < 50
        s = 50/quality;
    else
        s = 2-0.02*quality;
    end

    QY = [16    11    10    16    24    40    51    61;
          12    12    14    19    26    58    60    55;
          14    13    16    24    40    57    69    56;
          14    17    22    29    51    87    80    62;
          18    22    37    56    68   109   103    77;
          24    35    55    64    81   104   113    92;
          49    64    78    87   103   121   120   101;
          72    92    95    98   112   100   103    99];

    QY = round(s*QY);

    invQY = 1./QY;

    Y = yuv_comp_read(fileorig,'uint8',dims(2),dims(1),frames,frskip)-128;

    fct8x8(Y,Y); 
    Y = blockproc(Y,[8,8],@(block) round(bsxfun(@times,block.data,invQY)));

    %J = header;
    J.image_height = size(Y,1);
    J.image_width  = size(Y,2);
    J.coef_arrays{1} = Y;
    J.quant_tables{1} = QY;
    J.comp_info(1).quant_tbl_no = 1;
end

