function [ newDBN ] = dbninit( dbn, x )
%dbninit initilaze dbn using rbminit

for u = 1 : numel(dbn.sizes) - 1
    dbn.rbm{u} = rbminit( dbn.rbm{u}, x );
end
newDBN = dbn;

end

