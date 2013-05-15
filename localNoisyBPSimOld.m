function [err, errSet, W] = localNoisyBPSimOld(input,NumLayers, epsilon, gradStep, Tavg, numIter, randSeed)
%Input one (in future generalize to more inputs)

rng(randSeed) %seed random number generator

err = zeros(1,numIter);

M = size(input,1);
N= NumLayers;
T = Tavg;

numEx = size(input,2); %number of examples
errSet = zeros(numIter,numEx);

%numIter = 5000; (provided by input)- Note make this a maxIter and add a
%convergence threshold

layUp=[2:N]; %Set of layers to update
%Create non-linearity handle
g = @nonLin;

%Assuming only one signal 
%s = randn(M,1);

%Init one possible correct set of weights
Wsoln = (1/sqrt(M))*randn(M,M,N-1);

%Compute an output value the function can attain (at least with WCorr)
ySolnSet = propSig(1,N,Wsoln,input);

%Now we initialize the network
W = (1/sqrt(M))*randn(M,M,N-1);

%Initialize network of Neurons (for the whole time window)
x = zeros(M,N,T);

s = input(:,1);
x(:,1,1) = s;

noiseInit = epsilon*randn(M,N,T);

x = propNoisySig(x(:,:,1),s,W,noiseInit,T);

for cnt=1:numIter
    [cnt,numIter]
    
    exSet = randperm(numEx);
   
    for exCnt = exSet
        
        %Initialize parameters used to compute updates
        dW = zeros(size(W));
        dW_R = zeros(M,N-1);
        correl = zeros(M,M,N-1);
        %deltaX = zeros(M,1);
        
        s = input(:,exCnt);
        %s
        
        ySoln = ySolnSet(:,exCnt);
        %ySoln
        
        %propagate signal enough to remove old trace information
        for i=1:N
            x(:,:,T)=pNS1T(x(:,:,T),s,W,epsilon*randn(M,N));
        end
         
        %Run algorithm for T timesteps (and store all T)
        noise = epsilon*randn(M,N,T);
        x = propNoisySig(x(:,:,T),s,W,noise,T);

        deltaX = mean(repmat(ySoln,[1,1,T])- x(:,N,:),3);
        
        %exCnt
        %deltaX
              
        for c=layUp;
            tSet = 1:T-N+c;
            for t=tSet
                correl(:,:,c-1) = correl(:,:,c-1)+ x(:,N,t+(N-c))*noise(:,c,t)';
            end
            correl(:,:,c-1) = correl(:,:,c-1)/length(tSet); 
            
            dW_R(:,c-1) = correl(:,:,c-1)'*deltaX;
        end

        
        for c=layUp
            dW(:,:,c-1) = gradStep*dW_R(:,c-1)*mean(x(:,c-1,:),3)';
        end
        
        W= W+dW;   
        
        errSet(cnt, exCnt) = norm(deltaX)^2;
    end
        
end


err =sum(errSet,2);

end

