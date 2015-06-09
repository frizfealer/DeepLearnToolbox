function [ res ] = coherence( x, y )
%COHERENCE compute coherence of two vectors x, y
res = abs(x'*y) / ( norm(x)*norm(y) );

end

