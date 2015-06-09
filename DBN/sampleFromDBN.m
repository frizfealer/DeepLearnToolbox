function [ samples, insVec ] = sampleFromDBN( dbn, GibbIter, initMethod, samNum, varargin )
%sampleFromDBN generate a sample from dbn
netSize = length(dbn.rbm);
[hidNum, visNum] = size( dbn.rbm{end}.W );
if strcmp( initMethod, 'random' ) == 1
    cV1 = randn( 1, visNum )*1e-2;
%     cV1 = double( rand( samNum, visNum ) > 0.5 );
    cH1 = randn( 1, hidNum )*1e-2;
elseif strcmp( initMethod, 'sample' ) == 1
    cV1 = varargin{1};
    for i = 1:(netSize-1)
        cH1 = sigmrnd( dbn.rbm{i}.c' + cV1 * dbn.rbm{i}.W');
        cV1 = cH1;
    end
end
verbose = 1;
histVec = zeros(1000, 2);
for i = 1:GibbIter
    %         if i == 1
    %             cH1 = sigmrnd(repmat(dbn.rbm{end}.c', samNum, 1) + cV1 * dbn.rbm{end}.W');
    %         else
    cH1 = sigmrnd( dbn.rbm{end}.c' + cV1 * dbn.rbm{end}.W' );
    histVec(mod(i-1, 1000) + 1, 1) = cH1(1, 1);
    cV1 = sigmrnd( dbn.rbm{end}.b'+ cH1 * dbn.rbm{end}.W );
    histVec(mod(i-1, 1000) + 1, 2) = cV1(1, 1);
    if verbose == 1 && mod(i, 1000 ) == 1
        fprintf( 'mean of cH1 %g, mean of cV1 %g\n', mean(histVec(:, 1)), mean(histVec(:, 2)) );
    end
end
cStatus = zeros( samNum, dbn.sizes(end-1) );
fprintf( '%g %g\n', cV1 );
cnt = 1;
for i = 1:samNum*100
    cH1 = sigmrnd( dbn.rbm{end}.c' + cV1 * dbn.rbm{end}.W' );
    cV1 = sigmrnd( dbn.rbm{end}.b'+ cH1 * dbn.rbm{end}.W );
    if mod(i, 100) == 1
        cStatus(cnt, :) = cV1;
        cnt = cnt + 1;
    end
end
 ins = sum(cStatus,2);
 insVec = [length(find(ins==1)),  length(find(ins==2)),  length(find(ins==0)) ];
 fprintf( '%d %d %d\n',  length(find(ins==1)),  length(find(ins==2)),  length(find(ins==0)) );

for i = (netSize-1):-1:1
    nStatus = sigmrnd( repmat( dbn.rbm{i}.b', samNum, 1) + cStatus * dbn.rbm{i}.W );
    cStatus = nStatus;
end
samples = cStatus;
end

