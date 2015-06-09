function test_example_DBN
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex1 train a 100 hidden unit RBM and visualize its weights
% rand('state',0)
% dbn.sizes = [100];
% opts.numepochs =   50;
% opts.batchsize = 100;
% opts.momentum  =   0.5;
% opts.alpha     =  0.01;
% opts.type = 'bernoulli';
% opts.weightcost = 0.0002;
% opts.cohcost = 0.000;
% %InitialMomentumIter: initial momentum iteration using momentum
% %if larger then numepochs, trainning always uses momentum
% opts.InitialMomentumIter = 5;
% %After InitialMomentumIter, using FinalMomentum
% opts.FinalMomentum = 0.9;
% opts.cohSqcost = 0.000;
% opts.cohcost = 0.0;
% opts.projFlag = 1; opts.Temp = 10;
% opts.type = 'bernoulli';
% dbn = dbnsetup(dbn, train_x, opts);
% dbn = dbntrain(dbn, train_x, opts);
% figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rand('state',0)
%train dbn
dbn.sizes = [1000 1000 1000];
opts.numepochs =   100;
opts.batchsize = 100;
opts.momentum  =   0;
opts.InitialMomentumIter = 5;
%After InitialMomentumIter, using FinalMomentum
opts.FinalMomentum = 0;
opts.alpha     =   0.01;
opts.weightcost = 0.0000;
opts.cohcost = 0.000;
opts.type = 'bernoulli';
opts.Temp = 1;
opts.projFlag = 1;
opts.projTarget = 4;
opts.cohSqcost = 0;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
save( 'dbn.mat', 'dbn' );
%unfold dbn to nn
dropoutFraction = 0;
if dropoutFraction > 0
    for i = 1:length(dbn.rbm)
        dbn.rbm{i}.W = dbn.rbm{i}.W * (1/dropoutFraction);
    end
end
nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';
nn.learningRate = 0.1;
nn.weightPenaltyL2                  = 0;
nn.momentum = 0.5;
nn.projTarget = -1;
nn.name = 'dbn_proj-1';
nn.dropoutFraction = dropoutFraction;
%train nn
opts.numepochs =  200;
opts.batchsize = 100;
vx   = train_x(1:10000,:);
tx = train_x(10001:end,:);
vy   = train_y(1:10000,:);
ty = train_y(10001:end,:);
nn = nntrain(nn, tx, ty, opts, vx, vy);
[er, bad] = nntest(nn, test_x, test_y);
save( 'dbn_1.mat', 'dbn', 'nn' );
assert(er < 0.10, 'Too big error');
