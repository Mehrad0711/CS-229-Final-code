clear variables; clc;

% Features
Vars = {'AnkleDorsi_meanStance', 'AnkleDorsi_maxSwing', 'FootProg_meanStance'...
    , 'hasKinetics', 'age', 'speed', 'steplen', 'strideT', 'bmi', 'percentStanceSS'}; 

% Read Input Data
original_data = readtable('alldata.csv');

% Read Only One View 
data = original_data(char(original_data.side) == 'R', :);

% Target
cadence = data(:, 'cadence');

% Read Feature Values
data = data(:, Vars);

% Create Arrays from Tables
features = table2array(data);
cadence = table2array(cadence);

% Clean NAN Inputs
index_nan = isnan(features);
test = not(logical(sum(index_nan, 2)));
features = features(test, :);
cadence = cadence(test, :);

% Shuffle Data
rng(1);
[m, n] = size(features);
rand_ind = randperm(m);
features = features(rand_ind, :);
cadence = cadence(rand_ind);

% Choose What Fraction of Data is Needed
fraction = 1;
new_m = floor(size(features, 1) / fraction);
new_n = size(features, 2);
features = features(1:new_m, :);
cadence = cadence(1:new_m);

% Normalize Data
features  = features ./ max(features);
cadence = cadence ./ max(cadence);

% Divide Data With Ratio: [70, 15, 15]
train_features = features(1 : 0.7*new_m, :);
dev_features = features(0.7*new_m+1 : 0.85*new_m, :);
test_features = features(0.85*new_m+1 : new_m, :);

train_cadence = cadence(1 : 0.7*new_m);
dev_cadence = cadence(0.7*new_m+1 : 0.85*new_m);
test_cadence = cadence(0.85*new_m+1 : new_m);

% Write Output Files
csvwrite('train_features.csv',train_features);
csvwrite('dev_features.csv',dev_features);
csvwrite('test_features.csv',test_features);
csvwrite('train_cadence.csv',train_cadence);
csvwrite('dev_cadence.csv',dev_cadence);
csvwrite('test_cadence.csv',test_cadence);


