%Run tests on the different backpropagation algorithms over different
%classes of networks, etc. for a detailed comparison.

clear all;
close all;


%% Parameters to vary
dataDim=8;
numTrainEx =20;
numTestEx = 300;
numLayers=3;

Tavg=100;
numIter = 500;
stepSz=.025;

 
 
 
%In order tore-seed random number generators randomly each run
seed=randi(5000);
%seed = 7;
M=dataDim;
N= numLayers;

%rng(5)
%Init one possible correct set of weights
Wsoln = (1/sqrt(M))*randn(M,M,N-1);
input = randn(dataDim,numTrainEx);
 
 %To optimize, want to find the step size to converge the fastest (so need
 %to modify functions: numIter-> maxIter and find a tolerance at which you can stop)
 
 
[err,errSet, Wbp,WbpTime]=backprop(input,numLayers,stepSz,numIter,Wsoln, seed);
 
[err1, WlocBP, WlocBPTime] = localNoisyBPSim(input,numLayers,sqrt(stepSz),1,Tavg, numIter, Wsoln, seed);

subplot(1,2,1)
hold on;
plot([1:numIter],err)
plot([1:numIter],err1,'r--')

title(['Train Err w/ ',num2str(numTrainEx),' Examp, ',num2str(Tavg),' Tavg, and ',num2str(stepSz),' step size']);
ylabel('Error')
xlabel('Iteration')
legend('bp','bp local')





%now perform cross validation based on more examples


 
testInput = randn(dataDim,numTestEx);
 
ySolnSet = propSig(1,N,Wsoln,testInput);

errBP = zeros(1,numIter);
errLocBP = zeros(1,numIter);
for cnt=1:numIter
yBP = propSig(1,N,WbpTime(:,:,:,cnt),testInput);
errBP(cnt)=norm(yBP-ySolnSet,'fro')^2;

ylocBP = propSig(1,N,WlocBPTime(:,:,:,cnt),testInput);
errLocBP(cnt) = norm(ylocBP-ySolnSet,'fro')^2;

end

subplot(1,2,2)
hold on;
plot([1:numIter],errBP);
plot([1:numIter],errLocBP,'r--');

title(['Test Err w/ ',num2str(numTestEx),' Examp, ',num2str(Tavg),' Tavg, and ',num2str(stepSz),' step size, ', num2str(N),' layers, ', num2str(M),' dims']);
ylabel('Error')
xlabel('Iteration')
legend('bp','bp local')





%% Static training error
% numTestEx = 10;
%  
% testInput = randn(dataDim,numTestEx);
%  
% ySolnSet = propSig(1,N,Wsoln,testInput);

% yBP = propSig(1,N,Wbp,testInput);
% 
% disp('belief prop error')
% norm(yBP-ySolnSet)^2
% 
% ylocBP = propSig(1,N,WlocBP,testInput);
%  
%  disp('local bp error')
% norm(ylocBP-ySolnSet)^2


 
 
 %% TO DO: Set up bp's to output weights and run a non-noisy error test. This is just a check that everything is working

 %After outputting W's run a non-noisy test of generalization error? (check
 %if this is what Ben did)
 
 %Also can set eta to automatically update to optimize learning rate
 %(particularly if working on a bigger problem...)
 
