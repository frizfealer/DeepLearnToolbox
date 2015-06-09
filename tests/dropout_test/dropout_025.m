addpath(genpath('/nas02/home/y/h/yharn/DeepLearnToolbox'));
load mnist_uint8;
train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

load('dbn_2.mat');
%unfold dbn to nn
dropoutFraction = 0.25;
if dropoutFraction > 0
    for i = 1:length(dbn.rbm)
        dbn.rbm{i}.W = dbn.rbm{i}.W * (1/(1-dropoutFraction));
    end
end
nn = dbnunfoldtonn(dbn, 10);
nn.name = 'dbn_proj-1_do025';
nn.activation_function = 'sigm';
nn.learningRate = 0.1;
nn.weightPenaltyL2                  = 0;
nn.momentum = 0.5;
nn.projTarget = -1;
nn.dropoutFraction                  = dropoutFraction;
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
