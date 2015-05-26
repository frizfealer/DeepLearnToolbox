function [ log_base_rate ] = compute_logBaseRate( batchdata )
% Version 1.000
%
% Code provided by Ruslan Salakhutdinov, modified by YC Harn
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

%%%%% Initialize biases of the base-rate model by ML %%%%%%%%%%%%%%%%%%%%%
% Compute base rate with Wij, hidden biase = 0
% return biase terms for visible unit

[numcases, numdims, numbatches]=size(batchdata);
 count_int = zeros(numdims,1);
 for batch=1:numbatches
    xx = sum(batchdata(:,:,batch));
    count_int = count_int + xx';
 end

 lp=5;
 p_int = (count_int+lp*numbatches)/(numcases*numbatches+lp*numbatches);
 log_base_rate = log( p_int) - log(1-p_int);
end

