function dbn = dbnsetup(dbn, featureSize, opts)
%alpha: learning rate
%momentum: as the paper
%vW, vb, vc: velocity of W, b, c
%type: different type of distribution for visible unit
dbn.sizes = [featureSize, dbn.sizes];

for u = 1 : numel(dbn.sizes) - 1
    if u == 1
        dbn.rbm{u}.type = opts.type;
    else
        dbn.rbm{u}.type = 'bernoulli';
    end
    if isfield( opts, 'alpha' )
        dbn.rbm{u}.alpha    = opts.alpha;
    else
        dbn.rbm{u}.alpha = 0.1;
    end
    if isfield( opts, 'Temp' )
        dbn.rbm{u}.Temp    = opts.Temp;
    else
        dbn.rbm{u}.Temp = 1;
    end
    if isfield( opts, 'momentum' )
        dbn.rbm{u}.momentum    = opts.momentum;
    else
        dbn.rbm{u}.momentum = 0.5;
    end
    if isfield( opts, 'InitialMomentumIter' )
        dbn.rbm{u}.InitialMomentumIter    = opts.InitialMomentumIter;
    else
        dbn.rbm{u}.InitialMomentumIter = 5;
    end
    if isfield( opts, 'FinalMomentum' )
        dbn.rbm{u}.FinalMomentum    = opts.FinalMomentum;
    else
        dbn.rbm{u}.FinalMomentum = 0.9;
    end
    if isfield( opts, 'weightcost' )
        dbn.rbm{u}.weightcost    = opts.weightcost;
    else
        dbn.rbm{u}.weightcost = 0.0002;
    end
    if isfield( opts, 'cohSqcost' )
        dbn.rbm{u}.cohSqcost    = opts.cohSqcost;
    else
        dbn.rbm{u}.cohSqcost = 0;
    end
    if isfield( opts, 'cohcost' )
        dbn.rbm{u}.cohcost    = opts.cohcost;
    else
        dbn.rbm{u}.cohcost = 0;
    end
    if isfield( opts, 'projTarget' )
        dbn.rbm{u}.projTarget    = opts.projTarget;
    else
        dbn.rbm{u}.projTarget = -1;
    end
    
    dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u));
    dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));
    
    dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);
    dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);
    
    dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);
    dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
end

end
