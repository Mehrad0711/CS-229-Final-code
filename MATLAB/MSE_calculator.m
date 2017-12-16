
% Read Training Prediction and Actual Data
train_results = csvread('train_results.csv');
train_actual = csvread('train_cadence.csv');

n = size(train_results, 1);

% Calulate MSE
mse_train = sum((train_results - train_actual).^2) / n;

% Calculate # of sign differences
signs_train = sum (sign(train_results) - sign(train_actual));

% Regression Plot
figure(1);
title('train'); xlabel('target'); ylabel('prediction');
hold on;
scatter(train_actual, train_results);
fplot(@(x) x, [0 1])
hold off;

% Calculating Regression Parameters for Training
[r_train, m_train, b_train] = regression(train_actual', train_results');



% Read Test Prediction and Actual Data
test_results = csvread('test_results.csv');
test_actual = csvread('test_cadence.csv');

n = size(test_results, 1);

% Calulate MSE
mse_test = sum((test_results - test_actual).^2) / n;

% Calculate # of sign differences
signs_test = sum (sign(test_results) - sign(test_actual));

% Regression Plot
figure(2);
title('test'); xlabel('target'); ylabel('prediction');
hold on;
scatter(test_actual, test_results);
fplot(@(x) x, [0 1]);
hold off;

% Calculating Regression Parameters for Test
[r_test, m_test, b_test] = regression(test_actual', test_results');




