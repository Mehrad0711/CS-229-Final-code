% Returns a Trained Regression Model and its RMSE.

% Extract Predictors and Response
% Preprocessing Data for Regression
mat_whole_sim_reg;
inputTable = train_whole_sim_reg;
predictorNames = {'AnkleDorsi_meanStance', 'AnkleDorsi_maxSwing', 'FootProg_meanStance', 'hasKinetics', 'age', 'speed', 'steplen', 'strideT', 'bmi', 'percentStanceSS'};
predictors = inputTable(:, predictorNames);
response = inputTable.cadence;

% Train a regression model
% This code specifies all the model options and trains the model.
regressionTree = fitrtree(...
    predictors, ...
    response, ...
    'MinLeafSize', 12, ...
    'Surrogate', 'off');

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
treePredictFcn = @(x) predict(regressionTree, x);
trainedModel.predictFcn = @(x) treePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedModel.RequiredVariables = {'AnkleDorsi_meanStance', 'AnkleDorsi_maxSwing', 'FootProg_meanStance', 'hasKinetics', 'age', 'speed', 'steplen', 'strideT', 'bmi', 'percentStanceSS'};
trainedModel.RegressionTree = regressionTree;
trainedModel.About = 'This struct is a trained model exported from Regression Learner R2017b.';
trainedModel.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.');


% Perform cross-validation
partitionedModel = crossval(trainedModel.RegressionTree, 'KFold', 5);

% Compute validation predictions
validationPredictions = kfoldPredict(partitionedModel);

% Compute validation RMSE
validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));






% train error
train_data = train_whole_sim_reg;
train_actual_reg = train_data.cadence;

train_result_reg = trainedModel.predictFcn(train_data);

n = size(train_result_reg, 1);

Rmse_train = sqrt(sum((train_result_reg - train_actual_reg).^2) / n);

figure(1);
title('train'); xlabel('target'); ylabel('prediction');
hold on;
scatter(train_actual_reg, train_result_reg);
fplot(@(x) x, [0 1]);
hold off;

[r_train, m_train, b_train] = regression(train_actual_reg', train_result_reg');


% test error
test_data = test_whole_sim_reg;
test_actual_reg = test_data.cadence;


test_result_reg = trainedModel.predictFcn(test_data);

n = size(test_result_reg, 1);

Rmse_test = sqrt(sum((test_result_reg - test_actual_reg).^2) / n);

figure(2);
title('test'); xlabel('target'); ylabel('prediction');
hold on;
scatter(test_actual_reg, test_result_reg);
fplot(@(x) x, [0 1]);
hold off;

[r_test, m_test, b_test] = regression(test_actual_reg', test_result_reg');



