function nn = nnapplygrads(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    
    for i = 1 : (nn.n - 1)
        dW = nn.dW{i};
        if(nn.weightPenaltyL2>0)
            dW = dW + nn.weightPenaltyL2 * [zeros(size(nn.W{i},1),1) nn.W{i}(:,2:end)];
        end
        if(nn.cohSqcost>0)
            wVec = nn.W{i}(:,2:end);
            covTerm = wVec*wVec'*2;
            for z = 1:size(covTerm, 1)
                covTerm(z, z) = 0;
            end
            costTerm = covTerm*wVec;
            cohSqTerm = costTerm*nn.cohSqcost;
            dW(:,2:end) = dW(:,2:end) + cohSqTerm / size(nn.W{i}, 1);
        end
        if nn.cohcost > 0
            wVec = nn.W{i}(:,2:end);
            tmp2 = wVec; tmp3 = sum( tmp2, 1 );
            tmp2 = tmp3( ones( size(wVec, 1), 1 ), : ) - tmp2;
            dW(:,2:end) = dW(:,2:end) + tmp2 * nn.cohcost / size(nn.W{i}, 1);
        end
        
        dW = nn.learningRate * dW;
        
        if(nn.momentum>0)
            nn.vW{i} = nn.momentum*nn.vW{i} + dW;
            dW = nn.vW{i};
        end
            
        nn.W{i} = nn.W{i} - dW;
        if nn.projTarget ~= -1
            target = nn.projTarget;
            lenW = sqrt( sum( nn.W{i}(:,2:end).^2, 2 ) );
            lenW = max(lenW, 1);
            nn.W{i}(:,2:end)  = nn.W{i}(:,2:end) ./ lenW( :, ones(1, size(nn.W{i}, 2)-1 ) ) *target;
        end
    end
end
