function [ outEn ] = energyFunX( b, c, W, x )
%energyFunX compute energy of visible unit
hidNum = length(c);
vidNum = length(b);
outEn = 0;

%// Create all possible permutations (with repetition) of letters stored in x
M = permn([0 1], hidNum);
for i = 1:size(M, 1)
    outEn = outEn + exp( M(i, :)*W*x + b'*x + M(i, :)*c ); 
end
end

