function rbm = rbmtrain(rbm, x, opts)
    assert(isfloat(x), 'x must be a float');
    assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
    m = size(x, 1);
    numbatches = m / opts.batchsize;
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');
    hTemp = rbm.Temp;
    for i = 1 : opts.numepochs
        kk = randperm(m);
        err = 0;
        prbmW = rbm.W;
        for l = 1 : numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);
            
            v1 = batch;
            h1 = sigmrnd( hTemp * ( repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W' ) );
%             v2 = sigmrnd(repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W);
            v2 = sigm( hTemp * ( repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W ) );
            h2 = sigm( hTemp * ( repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W') );

            c1 = hTemp * h1' * v1;
            c2 = hTemp * h2' * v2;
            weightTerm = rbm.weightcost*rbm.W;
            if rbm.cohSqcost ~= 0
                wVec = rbm.W;
                covTerm = wVec*wVec'*2;
                for z = 1:size(covTerm, 1)
                    covTerm(z, z) = 0;
                end
                costTerm = covTerm*wVec;
                cohSqTerm = costTerm*rbm.cohSqcost;
            else
                cohSqTerm = 0;
            end
            if rbm.cohcost ~= 0
                tmp2 = rbm.W;
                tmp = zeros( size( rbm.W ) );
                tmp = repmat( sum( tmp2, 1 ), size(tmp, 1), 1) - tmp2;
                cohTerm = tmp * rbm.cohcost;
            else
                cohTerm = 0;
            end
            
            gradW = (c1 - c2) / opts.batchsize;
            
            rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * ( gradW - weightTerm - cohSqTerm - cohTerm );
            rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * hTemp*sum(v1 - v2)' / opts.batchsize;
            rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * hTemp*sum(h1 - h2)' / opts.batchsize;
            rbm.W = rbm.W + rbm.vW;
            if rbm.projTarget ~= -1
                target = rbm.projTarget;
                tmp = rbm.W.^2; tmp = sqrt(sum(tmp, 2));
                tmp = max(tmp, 1);
                rbm.W= rbm.W ./ repmat( tmp, 1, size(rbm.W, 2) ) *target;
            end
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc;
            err = err + sum(sum((v1 - double(v2>=0.5)) .^ 2)) / opts.batchsize;
        end
        
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
%         fprintf('length diff W:%g\n', norm(rbm.W(:)-prbmW(:)));
    end
%     figure; subplot(1,2,1); imagesc(v1); subplot(1,2,2); imagesc(v2);
end
