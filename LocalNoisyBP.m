%Try to do this computation the simplest way to begin, by performing 3
%seperate averages over some time interval T
clear all;
close all;


%Same non-linearity g used throughout
M = 5;   %Number of units per layer
N = 3; %Number of layers
T=500; %Number of timesteps to perform averaging over before recomputing weights
epsilon=.1; %standard dev of noise
gradStep = .1; %gradient stepSize

layUp=[2:N] %Set of layers to update
%Create non-linearity handle
g = @nonLin;

%Assuming only one signal 
s = randn(M,1);

%Init one possible correct set of weights
Wsoln = randn(M,M,N-1);

%Compute an output value the function can attain (at least with WCorr)
ySoln = propSig(1,N,Wsoln,s);

%Now we initialize the network
W = randn(M,M,N-1);

%Initialize network of Neurons (for the whole time window)
x = zeros(M,N,T);

x(:,1,1) = s;

noise = epsilon*randn(M,N,T);

x = propNoisySig(x(:,:,1),s,W,noise,T);

%Now average the appropriate quantities

numIter = 5000;

for cnt=1:numIter
    
    %Initialize parameters used to compute updates
    dW = zeros(size(W));
    dW_R = zeros(M,N-1);

    %Run algorithm for T timesteps (and store all T)
    noise = epsilon*randn(M,N,T);
    x = propNoisySig(x(:,:,N),s,W,noise,T);
    
    deltaX = mean(repmat(ySoln,[1,1,T])- x(:,N,:),3);
    norm(deltaX) %print average error from target
    
    
    %Run algorithm for T timesteps (and store all T)
    noise = epsilon*randn(M,N,T);
    x = propNoisySig(x(:,:,N),s,W,noise,T);

    correl = zeros(M,M,N-1);
    for c=layUp; %updating w_(c-1)
        
        tSet = 1:T-N+c;        
        for t=tSet
            correl(:,:,c-1) = correl(:,:,c-1)+ x(:,N,t+(N-c))*noise(:,c,t)';
        end
        correl(:,:,c-1) = correl(:,:,c-1)/length(tSet); 

        dW_R(:,c-1) = correl(:,:,c-1)'*deltaX;
    end
    
    %Run algorithm for T timesteps (and store all T)
    noise = epsilon*randn(M,N,T);
    x = propNoisySig(x(:,:,N),s,W,noise,T);
    
    for c=layUp
        dW(:,:,c-1) = gradStep*dW_R(:,c-1)*mean(x(:,c-1,:),3)';
    end
    
    W= W+dW;
    noise = epsilon*randn(M,N,T);
    x = propNoisySig(x(:,:,N),s,W,noise,T); %push effects of new weights to the end of the network
    
end

