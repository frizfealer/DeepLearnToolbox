function [ output_args ] = loadMNISTDBN( input_args )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%train dbn
dbn.sizes = [500 500 2000];
opts.numepochs =   50;
opts.batchsize = 100;
opts.momentum  =   0.5;
opts.InitialMomentumIter = 5;
%After InitialMomentumIter, using FinalMomentum
opts.FinalMomentum = 0.9;
opts.alpha     =   0.1;
opts.weightcost = 0.0002;
opts.cohcost = 0.000;
opts.cohSqcost = 0.000;
opts.projFlag = 1; opts.Temp = 10;
opts.type = 'bernoulli';
load( 'mnist_trainData.mat' );
load( 'mnist_trainLabel.mat' );
load( 'mnist_testData.mat' );
load( 'mnist_testLabel.mat' );
% dummy = sparse(6000, 784);
% dbn = dbnsetup(dbn, dummy, opts);
% dbn.rbm{1}.W = vishid'; dbn.rbm{1}.b = visbiases'; dbn.rbm{1}.c = hidrecbiases'; 
% dbn.rbm{2}.W = hidpen'; dbn.rbm{2}.b = hidgenbiases'; dbn.rbm{2}.c = penrecbiases'; 
% dbn.rbm{3}.W = hidpen2'; dbn.rbm{3}.b = hidgenbiases2'; dbn.rbm{3}.c = penrecbiases2';
load( 'mnist_pretrain_DBN.mat' );
%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';
%train nn
opts.numepochs =  200;
opts.batchsize = 100;
nn.learningRate = 1;                %  Sigm require a lower learning rate
vx   = trainData(1:10000,:);
tx = trainData(10001:end,:);
vy   = trainLabel(1:10000,:);
ty = trainLabel(10001:end,:);
nn = nntrain(nn, tx, ty, opts, vx, vy);

[er, bad] = nntest(nn, testData, test_Label);

end

