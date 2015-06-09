function [ avgConstuctErr, cV1 ] = rbmtest( rbm, testData, verbose )
%rbmtest run rbm model on a testing data set. 
%Currently only support Bernoulli distribution.
GibbIter = 1;
cV1 = testData;
samNum = size( testData, 1);
cH1 = zeros( samNum, length(rbm.c) );
hTemp = rbm.Temp;
for i = 1:GibbIter
    pH = cH1;
    pV = cV1;
    if i == 1
        cH1 = sigmrnd( hTemp * ( repmat(rbm.c', samNum, 1) + cV1 * rbm.W' ) );
    else
        cH1 = sigm( hTemp * ( repmat(rbm.c', samNum, 1) + cV1 * rbm.W' ) );
    end
    cV1 = sigm( hTemp * ( repmat(rbm.b', samNum, 1) + cH1 * rbm.W ) );
    %fprintf( 'reconstruct error: %g\n', r2ErrVec(i) );
    diffH = norm(pH-cH1);
    diffV = norm(pV-cV1);
    if verbose == 1
        fprintf( 'iter %d: H difference: %g\t V difference: %g\n', i, diffH, diffV );
    end
    if diffH < 1e-6 && diffV < 1e-6
        break;
    end
end
cV1 = double( cV1 > 0.5 );
avgConstuctErr = sum( sum( ( cV1 - testData ).^2 ) ) / samNum;

end

