%% first clear and close everything
clear all;
close all;
clc
%% create, configure, and initialize
load bodyfat_dataset;
net = feedforwardnet([10,10]);   % def: 2 layers (1 hidden, 1 output), 10 units
% configure the net: initialize weight & biases
net = configure(net, bodyfatInputs, bodyfatTargets); 
% train() func also configures the net
view(net); % view the graphical representation

% Train and Apply Multilayer Shallow Neural Networks
% e.g., for function approximation (nonlinear regression) or pattern recognition
[net,tr] = train(net,bodyfatInputs,bodyfatTargets);
net.trainParam.min_grad=0.001;
%% Use the network
% the network response to the fifth input vector in the building dataset
est = net(bodyfatInputs(:,5))
actual = bodyfatTargets(5)
% check all - batch mode: more efficient
all = net(bodyfatInputs);

%% Analyze performance after training
tr
plotperf(tr);

% Regression analysis 
% The first command calculates the trained network response to all of the 
% inputs in the data set. 
% The following six commands extract the outputs and targets that belong to
% the training, validation and test subsets. 
% The final command creates three regression plots for training, testing 
% and validation.
bodyfatOutputs = net(bodyfatInputs);
trOut = bodyfatOutputs(tr.trainInd);
vOut = bodyfatOutputs(tr.valInd);
tsOut = bodyfatOutputs(tr.testInd);
trTarg = bodyfatTargets(tr.trainInd);
vTarg = bodyfatTargets(tr.valInd);
tsTarg = bodyfatTargets(tr.testInd);
plotregression(trTarg, trOut, 'Train', vTarg, vOut, 'Validation', tsTarg, tsOut, 'Testing')


%% Improving results
%% maybe a different initialization? ideal: check n-many initialization for
% consistency.
net = init(net);
[net,tr]  = train(net, bodyfatInputs, bodyfatTargets);

%% a different network architecture! but, be aware of bias-variance trade-off
net = feedforwardnet([10 10]);   % def: 3 layers (2 hidden, 1 out), 10u/l
net = init(net);
view(net)
[net,tr]  = train(net, bodyfatInputs, bodyfatTargets);

%% a different training function!
net = feedforwardnet([20 20]);   % def: 3 layers (2 hidden, 1 out), 10u/l
net = init(net);
net.trainFcn = 'trainscg'         % Bayesian regularization 
view(net)
[net,tr]  = train(net, bodyfatInputs, bodyfatTargets);

% add more training data - good luck!

