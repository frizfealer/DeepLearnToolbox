function coherence_toy_1( exp_Num )
dbn.sizes = [2 2];
feature_size = 9;
opts.numepochs =   100;
opts.batchsize = 100;
opts.momentum  =   0;
opts.InitialMomentumIter = 5;
%After InitialMomentumIter, using FinalMomentum
opts.FinalMomentum = 0;
opts.alpha     =   0.1;
opts.weightcost = 0.0000;
opts.cohcost = 0.000;
opts.type = 'bernoulli';
opts.Temp = 1;
opts.projFlag = 0;
opts.projTarget = 4;
opts.cohSqcost = 0;
dbn = dbnsetup(dbn, feature_size, opts);

%cross filter
crossPat1 = [-4 4 -4; 4 4 4; -4 4 -4];
crossPat2 = [4 -4 4; -4 4 -4; 4 -4 4];

dbn.rbm{1}.W(1, :) = crossPat1(:)*8;
dbn.rbm{1}.W(2, :) = crossPat2(:)*8;
dbn.rbm{1}.b = ones(9, 1)*-8;
dbn.rbm{1}.c = ones(2, 1)*0;
%experiment 1:
%with first type of in-coherence: [1 0]', [0 1]'
if ~isempty( find( exp_Num == 1, 1 ) )
    dbn.rbm{2}.W(1, :) = [2 0];
    dbn.rbm{2}.W(2, :) = [0 2];
    dbn.rbm{2}.b = ones(2, 1)*0;
    dbn.rbm{2}.c = ones(2, 1)*0;
    resultVec = zeros( 1000, 3 );
    [samples, insVec] = sampleFromDBN( dbn, 1000, 'random', 100 );
    figure;
    M = permn([0 1], length(dbn.rbm{end}.b));
    outEn = zeros(size(M, 1), 1);
    for i = 1:size(M, 1)
        [ outEn(i) ] = energyFunX( dbn.rbm{end}.b, dbn.rbm{end}.c, dbn.rbm{end}.W, M(i, :)');
    end
end
%experiment 2:
%with first type of in-coherence: [1 -1]', [-1 1]'
%and sample initialization
if ~isempty( find( exp_Num == 2, 1 ) )
    dbn.rbm{2}.W(1, :) = [4 -4];
    dbn.rbm{2}.W(2, :) = [-4 4];
    dbn.rbm{2}.b = ones(2, 1)*0;
    dbn.rbm{2}.c = ones(2, 1)*0;
    [ samples, insVec] = sampleFromDBN( dbn, 1000, 'random', 1000 );
    M = permn([0 1], length(dbn.rbm{end}.b));
    outEn = zeros(size(M, 1), 1);
    for i = 1:size(M, 1)
        [ outEn(i) ] = energyFunX( dbn.rbm{end}.b, dbn.rbm{end}.c, dbn.rbm{end}.W, M(i, :)');
    end
end

%experiment 3:
%with first type of in-coherence: [1 -1]', [1 1]'
%and sample initialization
if ~isempty( find( exp_Num == 3, 1 ) )
    dbn.rbm{2}.W(1, :) = [1 1];
    dbn.rbm{2}.W(2, :) = [1 0];
    dbn.rbm{2}.b = ones(2, 1)*0;
    dbn.rbm{2}.c = ones(2, 1)*0;
    [ samples, insVec] = sampleFromDBN( dbn, 1000, 'random', 1000 );
    M = permn([0 1], length(dbn.rbm{end}.b));
    outEn = zeros(size(M, 1), 1);
    for i = 1:size(M, 1)
        [ outEn(i) ] = energyFunX( dbn.rbm{end}.b, dbn.rbm{end}.c, dbn.rbm{end}.W, M(i, :)');
    end
end

dbn.sizes = [2 1];
feature_size = 9;
opts.numepochs =   100;
opts.batchsize = 100;
opts.momentum  =   0;
opts.InitialMomentumIter = 5;
%After InitialMomentumIter, using FinalMomentum
opts.FinalMomentum = 0;
opts.alpha     =   0.1;
opts.weightcost = 0.0000;
opts.cohcost = 0.000;
opts.type = 'bernoulli';
opts.Temp = 1;
opts.projFlag = 0;
opts.projTarget = 4;
opts.cohSqcost = 0;
dbn = dbnsetup(dbn, feature_size, opts);

%experiment 4:
%with first type of in-coherence: [1 -1]', [1 1]'
%and sample initialization
if ~isempty( find( exp_Num == 4, 1 ) )
    dbn.rbm{2}.W(1, :) = [1 1];
    dbn.rbm{2}.b = ones(2, 1)*0;
    dbn.rbm{2}.c = ones(1, 1)*0;
    [ samples, insVec] = sampleFromDBN( dbn, 1000, 'random', 100 );
    M = permn([0 1], length(dbn.rbm{end}.b));
    outEn = zeros(size(M, 1), 1);
    for i = 1:size(M, 1)
        [ outEn(i) ] = energyFunX( dbn.rbm{end}.b, dbn.rbm{end}.c, dbn.rbm{end}.W, M(i, :)');
    end
end


end