clear;
%% training data
x_train = linspace(-1,1,10);
y_train = linspace(-1,1,10);
[X_train,Y_train] = meshgrid(x_train,y_train);
Z_train=cos(X_train + 6 *0.35*Y_train)+2 *0.35* X_train.*Y_train;
Input_train = [X_train(:)';Y_train(:)'];
Output_train = Z_train(:)';
%surf(X_train,Y_train,Z_train)
%view(3)

%% validation data
x_val = random('Uniform',-1,1,[1,7]);
y_val = random('Uniform',-1,1,[1,7]);
[X_val,Y_val] = meshgrid(x_val,y_val);
Z_val = cos(X_val + 6 *0.35*Y_val)+2 *0.35* X_val.*Y_val;
Input_val = [X_val(:)';Y_val(:)'];
Output_val = Z_val(:)';
figure(4);
surf(X_val,Y_val,Z_val);
view(3)
%% test data
x_test = linspace(-1,1,9);
y_test = linspace(-1,1,9);
[X_test,Y_test] = meshgrid(x_test,y_test);
Z_test = cos(X_test + 6 *0.35*Y_test)+2 *0.35* X_test.*Y_test;
Input_test = [X_test(:)';Y_test(:)'];
Output_test = Z_test(:)';
figure(3);
surf(X_test,Y_test,Z_test)
view(3)
%% network design
Input = [Input_train Input_val Input_test];
Output = [Output_train Output_val Output_test];
% net = feedforwardnet (); %def:2layers (1hidden,1 output)
num_hidden = 8;  % 2 8 50
%TrainFcn = 'trainbfg';
TrainFcn = 'traingd';
%TrainFcn = 'traingdm';
%TrainFcn = 'trainlm';

net = feedforwardnet([num_hidden],TrainFcn);

% configure the net: initialize weight & bias 
net = configure(net, Input, Output);
% train() func also configures the net
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';
view(net);

%options = trainingOptions('sgdm','InitialLearnRate',0.02,'LearnRateDropFactor',0.0);
% net.trainParam.min_grad < 0.00000000000000000001;
net.trainParam.epochs = 100;
% net.performFcn = 'mse';
%net.performParam.regularization = 0.02;
% net.trainParam.max_fail = net.trainParam.epochs;
net.divideParam.trainRatio = size(Input_train) / size(Input);
net.divideParam.valRatio = size(Input_val) / size(Input);
net.divideParam.testRatio = size(Input_test) / size(Input);
net.trainParam.alpha = 0.02;
[net,tr] = train(net, Input_train, Output_train);

%%
%evaluation
est= net(Input_test);
actual = Output_test;
%%
figure(1);
surf(X_test,Y_test,Z_test);
view(3)

figure(2);
est_matrix = reshape(est', size(Z_test));
surf(X_test,Y_test, est_matrix);
view(3)

%%
figure(5);
postreg(est,actual);
