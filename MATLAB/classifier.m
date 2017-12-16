% Returns a Trained Classification Model and its RMSE.

% Extract Predictors and Response
% Preprocessing Data for Classification
gmfcs;
inputTable = gmfcs_whole_sim_class;
predictorNames = {'Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6', 'Var7', 'Var8', 'Var9', 'Var10', 'Var11', 'Var12', 'Var13', 'Var14', 'Var15', 'Var16', 'Var17', 'Var18', 'Var19', 'Var20', 'Var21', 'Var22', 'Var23', 'Var24', 'Var25', 'Var26', 'Var27', 'Var28', 'Var29', 'Var30', 'Var31', 'Var32', 'Var33', 'Var34', 'Var35', 'Var36', 'Var37', 'Var38', 'Var39', 'Var40', 'Var41', 'Var42', 'Var43', 'Var44', 'Var45', 'Var46', 'Var47', 'Var48', 'Var49', 'Var50', 'Var51', 'Var52', 'Var53', 'Var54', 'Var55', 'Var56', 'Var57', 'Var58', 'Var59', 'Var60', 'Var61', 'Var62', 'Var63', 'Var64', 'Var65', 'Var66', 'Var67', 'Var68', 'Var69', 'Var70', 'Var71', 'Var72', 'Var73', 'Var74', 'Var75', 'Var76', 'Var77', 'Var78', 'Var79', 'Var80', 'Var81', 'Var82', 'Var83', 'Var84', 'Var85', 'Var86', 'Var87', 'Var88', 'Var89', 'Var90', 'Var91', 'Var92', 'Var93', 'Var94', 'Var95', 'Var96', 'Var97', 'Var98', 'Var99', 'Var100', 'Var101', 'Var102', 'Var103', 'Var104', 'Var105', 'Var106', 'Var107', 'Var108', 'Var109', 'Var110', 'Var111', 'Var112', 'Var113', 'Var114', 'Var115', 'Var116', 'Var117', 'Var118', 'Var119', 'Var120', 'Var121', 'Var122', 'Var123', 'Var124', 'Var125', 'Var126', 'Var127', 'Var128', 'Var129', 'Var130', 'Var131', 'Var132', 'Var133', 'Var134', 'Var135', 'Var136', 'Var137', 'Var138', 'Var139', 'Var140', 'Var141', 'Var142', 'Var143', 'Var144', 'Var145', 'Var146', 'Var147', 'Var148', 'Var149', 'Var150', 'Var151', 'Var152', 'Var153', 'Var154', 'Var155', 'Var156', 'Var157', 'Var158', 'Var159', 'Var160', 'Var161', 'Var162', 'Var163', 'Var164', 'Var165', 'Var166', 'Var167', 'Var168', 'Var169', 'Var170', 'Var171', 'Var172', 'Var173', 'Var174', 'Var175', 'Var176', 'Var177', 'Var178', 'Var179', 'Var180', 'Var181', 'Var182', 'Var183', 'Var184', 'Var185', 'Var186', 'Var187', 'Var188', 'Var189', 'Var190', 'Var191', 'Var192', 'Var193', 'Var194', 'Var195', 'Var196', 'Var197', 'Var198', 'Var199', 'Var200', 'Var201', 'Var202', 'Var203', 'Var204', 'Var205', 'Var206', 'Var207', 'Var208', 'Var209', 'Var210', 'Var211', 'Var212', 'Var213', 'Var214', 'Var215', 'Var216', 'Var217', 'Var218', 'Var219', 'Var220', 'Var221', 'Var222', 'Var223', 'Var224', 'Var225', 'Var226', 'Var227', 'Var228', 'Var229', 'Var230', 'Var231', 'Var232', 'Var233'};
predictors = inputTable(:, predictorNames);
response = inputTable.Var234;

% Train Classifier
template = templateTree(...
    'MaxNumSplits', 20);
classificationEnsemble = fitcensemble(...
    predictors, ...
    response, ...
    'Method', 'RUSBoost', ...
    'NumLearningCycles', 30, ...
    'Learners', template, ...
    'LearnRate', 0.1, ...
    'ClassNames', [1; 2; 3]);

% Create the result struct with predict function
predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedClassifier.RequiredVariables = {'Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6', 'Var7', 'Var8', 'Var9', 'Var10', 'Var11', 'Var12', 'Var13', 'Var14', 'Var15', 'Var16', 'Var17', 'Var18', 'Var19', 'Var20', 'Var21', 'Var22', 'Var23', 'Var24', 'Var25', 'Var26', 'Var27', 'Var28', 'Var29', 'Var30', 'Var31', 'Var32', 'Var33', 'Var34', 'Var35', 'Var36', 'Var37', 'Var38', 'Var39', 'Var40', 'Var41', 'Var42', 'Var43', 'Var44', 'Var45', 'Var46', 'Var47', 'Var48', 'Var49', 'Var50', 'Var51', 'Var52', 'Var53', 'Var54', 'Var55', 'Var56', 'Var57', 'Var58', 'Var59', 'Var60', 'Var61', 'Var62', 'Var63', 'Var64', 'Var65', 'Var66', 'Var67', 'Var68', 'Var69', 'Var70', 'Var71', 'Var72', 'Var73', 'Var74', 'Var75', 'Var76', 'Var77', 'Var78', 'Var79', 'Var80', 'Var81', 'Var82', 'Var83', 'Var84', 'Var85', 'Var86', 'Var87', 'Var88', 'Var89', 'Var90', 'Var91', 'Var92', 'Var93', 'Var94', 'Var95', 'Var96', 'Var97', 'Var98', 'Var99', 'Var100', 'Var101', 'Var102', 'Var103', 'Var104', 'Var105', 'Var106', 'Var107', 'Var108', 'Var109', 'Var110', 'Var111', 'Var112', 'Var113', 'Var114', 'Var115', 'Var116', 'Var117', 'Var118', 'Var119', 'Var120', 'Var121', 'Var122', 'Var123', 'Var124', 'Var125', 'Var126', 'Var127', 'Var128', 'Var129', 'Var130', 'Var131', 'Var132', 'Var133', 'Var134', 'Var135', 'Var136', 'Var137', 'Var138', 'Var139', 'Var140', 'Var141', 'Var142', 'Var143', 'Var144', 'Var145', 'Var146', 'Var147', 'Var148', 'Var149', 'Var150', 'Var151', 'Var152', 'Var153', 'Var154', 'Var155', 'Var156', 'Var157', 'Var158', 'Var159', 'Var160', 'Var161', 'Var162', 'Var163', 'Var164', 'Var165', 'Var166', 'Var167', 'Var168', 'Var169', 'Var170', 'Var171', 'Var172', 'Var173', 'Var174', 'Var175', 'Var176', 'Var177', 'Var178', 'Var179', 'Var180', 'Var181', 'Var182', 'Var183', 'Var184', 'Var185', 'Var186', 'Var187', 'Var188', 'Var189', 'Var190', 'Var191', 'Var192', 'Var193', 'Var194', 'Var195', 'Var196', 'Var197', 'Var198', 'Var199', 'Var200', 'Var201', 'Var202', 'Var203', 'Var204', 'Var205', 'Var206', 'Var207', 'Var208', 'Var209', 'Var210', 'Var211', 'Var212', 'Var213', 'Var214', 'Var215', 'Var216', 'Var217', 'Var218', 'Var219', 'Var220', 'Var221', 'Var222', 'Var223', 'Var224', 'Var225', 'Var226', 'Var227', 'Var228', 'Var229', 'Var230', 'Var231', 'Var232', 'Var233'};
trainedClassifier.ClassificationEnsemble = classificationEnsemble;

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 5);

% Compute validation predictions
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');


% Read Training Data
train_data = table2array(csvread('gmfcs_train_whole_sim_class.csv'));
train_actual_class = train_data.Var234;

% Predict Class Value
train_result_class = trainedClassifier.predictFcn(train_data);
n = size(train_result_class, 1);

% Accuracy
accuracy_train = 1 - sqrt(sum((train_result_class - train_actual_class).^2) / n);

% Confusion Matrix
C_train = confusionmat(train_actual_class, train_result_class);

% Ratio of Correctly Classified Data
aggregated_accu_train = trace(C_train) / sum(C_train(:));

% Calculate Regression Parameters
[r_train, m_train, b_train] = regression(train_actual_class', train_result_class');


% Read Test Data
test_data = table2array(csvread('gmfcs_test_whole_sim_class.csv'));
test_actual_class = test_data.Var234;

% Predict Class Value
test_result_class = trainedClassifier.predictFcn(test_data);
n = size(test_result_class, 1);

% Accuracy
accuracy_test = 1 - sqrt(sum((test_result_class - test_actual_class).^2) / n);

% Confusion Matrix
C_test = confusionmat(test_actual_class, test_result_class);

% Ratio of Correctly Classified Data
aggregated_accu_test = trace(C_test) / sum(C_test(:));

% Calculate Regression Parameters
[r_test, m_test, b_test] = regression(test_actual_class', test_result_class');

