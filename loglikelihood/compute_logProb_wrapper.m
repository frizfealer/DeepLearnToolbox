function [ loglik_test_est ] = compute_logProb_wrapper( vishid,hidbiases,visbiases, batchdata, testbatchdata, verbose )
%compute_logProb_wrapper a wrapper function to compute the log likelihood
%from a RBM model
%vishid: weight from visible units to hidden units.
%hidbiases: biases of hidden units.
%visbiases: biases of visible units.
%batchdata: data for trainning RBM model
%testbatchbata: data for testing.
%verbose: if equal to 1 print beta value, output figures.
[numcases, ~, numbatches]=size(testbatchdata);

%%%%% Estimate Test Log-Probability
 data = [];
 for ii=1:numbatches
   data = [data; testbatchdata(:,:,ii)];
 end

beta = [0:1/1000:0.5 0.5:1/10000:0.9 0.9:1/100000:1.0];
numruns = numcases;
rand('state',30);
randn('state',30);

% verbose = 1;
[logZZ_est, logZZ_est_up, logZZ_est_down] = ...
             compute_RBM_AIS(vishid,hidbiases,visbiases,numruns,beta,batchdata, verbose);

loglik_test_est = calculate_logprob(vishid,hidbiases,visbiases,logZZ_est,testbatchdata);
 
 
end

