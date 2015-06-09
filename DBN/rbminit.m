function [ initRBM ] = rbminit( rbm, X )
%rbminit initialize weight and bias terms according to the suggestion of "A
%Practical Guide to Training Restricted Boltzmann Machines".
%A return value function
%rbm: a rbm structure from dbnsetup
%X: an input matrix with row equals to data and column equals to features.
rbm.W = randn( size( rbm.W ) )*0.01;
rbm.c = zeros( size( rbm.c ) );
if strcmp( rbm.type, 'gaussian' ) == 1
    ind = X > 0.5;
    ind = sum( ind, 1 );
    proportion = ind / dim(X, 1);
    rbm.b = log( proportion ./ (1- proportion) )';
%     rbm.sigma = ones( dim(X, 2), 1 );
elseif strcmp( rbm.type, 'bernoulli' ) == 1
    ind = X == 1;
    ind = sum( ind, 1);
    proportion = ind / size( X, 1 );
    rbm.b = log( proportion ./ (1- proportion) )';
end
initRBM = rbm;

end

