% Run the Script File to Generate Data
cs229;

% Read Data
x = train_features';
t = train_cadence';

% Choose Training Function
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a 2-layered Fitting Network
hiddenLayerSize = [10, 10];
net = fitnet(hiddenLayerSize, trainFcn);

% Normalize Each Feature (Mapping to Range = [-1 1]) 
net.input.processFcns = {'mapminmax'};
net.output.processFcns = {'mapminmax'};

% Setup Division of Data for Training, Validation, Testing = [70, 15, 15]
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Choose MSE as Performance Function
net.performFcn = 'mse';

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
% net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
%     'plotregression', 'plotfit'};

% Train the Network
[net, tr] = train(net, x, t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y);
valPerformance = perform(net,valTargets,y);
testPerformance = perform(net,testTargets,y);

% View the Network
% view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(t,y)
%figure, plotfit(net,x,t)

