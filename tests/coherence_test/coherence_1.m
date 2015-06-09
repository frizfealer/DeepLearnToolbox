addpath(genpath('/nas02/home/y/h/yharn/DeepLearnToolbox'));
load mnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

load('dbn_2.mat');
%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 10);
nn.name = 'dbn_proj-1_coh01';
nn.activation_function = 'sigm';
nn.learningRate = 0.1;
nn.weightPenaltyL2                  = 0;
nn.momentum = 0.5;
nn.projTarget = -1;
nn.dropoutFraction                  = 0;
nn.cohSqcost = 1; %1 should be the maximun
%train nn
opts.numepochs =  200;
opts.batchsize = 100;
vx   = train_x(1:10000,:);
tx = train_x(10001:end,:);
vy   = train_y(1:10000,:);
ty = train_y(10001:end,:);
nn = nntrain(nn, tx, ty, opts, vx, vy);
[er, bad] = nntest(nn, test_x, test_y);
save( [nn.name, '.mat'], 'dbn', 'nn' );
