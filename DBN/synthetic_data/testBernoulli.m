function [ savedX ] = testBernoulli( savedX )
%testBernoulli A script to generate synthetic example to test coherence on
%RBM.
%% generate data
if isempty( savedX )
    X = double( rand( 1000, 4 ) > 0.5 );
    X = [X X(:, 1:2)];
    X = [X double( X(:, 3:4) == 0 )];
    savedX = X;
else
    X = savedX;
end
trainX = X(1:500, :);
testX = X(501:end,:);
%% setup RBM
rand('state',0)
dbn.sizes = [6];
opts.numepochs =   200;
opts.batchsize = 10;
opts.momentum  =   0.5;
opts.weightcost = 0.0002;
opts.cohSqcost = 0.001; %cohernece square penalty
opts.cohcost = 0.0; %coherence without absolute value penalty, need to use with projTarget, Temp
% opts.projFlag = 1; opts.Temp = 10;
%InitialMomentumIter: initial momentum iteration using momentum
opts.InitialMomentumIter = 5;
%After InitialMomentumIter, using FinalMomentum
opts.FinalMomentum = 0.9;
opts.alpha     =   0.1;
opts.type = 'bernoulli'; %currently only bernoulli
opts.projTarget = -1; %project length of weight of each hidden unit
opts.Temp = 1; %temperature in RBM
%% trainning
feature_size = 8;
dbn = dbnsetup(dbn, feature_size, opts);
dbn = dbninit( dbn, trainX );
dbn = dbntrain(dbn, trainX, opts, [], []);
%% show the results
verbose = 0;
[ avgConstuctErr, cV1 ] = rbmtest( dbn.rbm{1}, testX, verbose );
figure; imagesc(cV1-testX); colorbar; title(['difference of testX and the reconstructions. avgConstructErr = ', num2str(avgConstuctErr)]);
figure; subplot(1, 2, 1); imagesc(dbn.rbm{1}.W); title( 'W'); xlabel( 'features' ); ylabel(' h-unit' ); colorbar;
ins = dbn.rbm{1}.W; 
for i = 1:size(ins, 1)
    ins(i, :) = ins(i, :) / norm(ins(i, :));
end
ins = ins*ins';
subplot(1, 2, 2); imagesc(abs(ins)), title( 'coherence of W' ); xlabel( 'h-unit' ); ylabel( 'h-unit' ); colorbar;
h1 = sigm(repmat(dbn.rbm{1}.c', size(trainX, 1), 1) + trainX * dbn.rbm{1}.W');
figure; subplot(1, 2, 1); imagesc(h1); title( 'hidden prob.' ); colorbar; xlabel( 'h-unit' ); ylabel( 'samples' );
subplot(1, 2, 2); imagesc(h1>0.5); title( 'hidden fire?' ); xlabel( 'hunit' ); ylabel( 'samples' );
ins = sum(double(h1>0.5), 1) / size( trainX, 1 );
ins2 = (double(h1>0.5));
for i = 1:size(ins2, 2)
    ins2(:, i) = ins2(:, i) / norm( ins2(:, i) );
end
ins2 = ins2'*ins2;
figure; subplot(1, 2, 1); plot(ins); title('firing ratio of hunits' ); xlabel( 'h-units' ); ylabel( 'firing ratio' );
subplot(1, 2, 2); imagesc(abs(ins2)); title('firing coherence'); colorbar; xlabel( 'h-units' ); ylabel( 'h-units' );
end

