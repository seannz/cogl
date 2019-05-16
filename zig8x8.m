function z = zig8x8
    A = fliplr(reshape(1:64,8,8));
    z = zeros(64,1);
    for k =  7:-1:1
        p = ((7-k)^2+(7-k))/2;
        if mod(k,2) == 0
            z(p+(1:+1:8-k)) = diag(A,k);
        else
            z(p+(8-k:-1:1)) = diag(A,k);
        end
    end
    for k = -7:+1:0
        p = -((7+k)^2+(7+k))/2 + 65;
        if mod(k,2) == 1
            z(p-(1:+1:8+k)) = diag(A,k);
        else
            z(p-(8+k:-1:1)) = diag(A,k);
        end
    end
end