function y = rgb2ycc(x)
    x = double(x);
    [m,n,p] = size(x);

    Kb = 0.114;
    Kr = 0.299;
    Pb = 2 * (1-Kb);
    Pr = 2 * (1-Kr);
    
    T =  [Kr   , (1-Kr-Kb)   , Kb    ;
         -Kr/Pb,-(1-Kr-Kb)/Pb, 0.500 ;
          0.500,-(1-Kr-Kb)/Pr,-Kb/Pr];    

    y = reshape(reshape(x, m*n, 3) * T', m, n, p);
    %y = y + 128;
    y(:,:,2:3) = y(:,:,2:3) + 128;
end