function dbn = dbntrain(dbn, x, opts, startFromLayer, tmpFileName)
    n = numel(dbn.rbm);
    if isempty( tmpFileName )
        tmpFileFlag = 0; tmpFileName = [];
    else
        tmpFileFlag = 1; tmpFileName = 'dbn_tmp.mat';
    end
    if isempty( startFromLayer )
        startFromLayer = 1;
    end
    assert( startFromLayer <= n );
    if startFromLayer == 1
        dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, opts);
        if tmpFileFlag == 1
            save( tmpFileName, 'dbn' );
        end
        for i = 2 : n
            x = rbmup(dbn.rbm{i - 1}, x);
            dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, opts);
            if tmpFileFlag == 1
                save( tmpFileName, 'dbn' );
            end
        end
    else
        for i = startFromLayer : n
            x = rbmup(dbn.rbm{i - 1}, x);
            dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, opts);
            if tmpFileFlag == 1
                save( tmpFileName, 'dbn' );
            end
        end
    end

end
