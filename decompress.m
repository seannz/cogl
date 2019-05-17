function Y = decompress(J,c)
    Y = blockproc(J.coef_arrays{c},[8,8],@(block) bsxfun(@times,block.data,J.quant_tables{J.comp_info(c).quant_tbl_no}));
    ifct8x8(Y,Y);
end