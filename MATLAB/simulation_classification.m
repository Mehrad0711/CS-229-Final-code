clear variables; clc;

% Features + Target
Vars = {'AnkleDorsi_meanStance', 'AnkleDorsi_maxSwing', 'FootProg_meanStance'...
    , 'hasKinetics', 'age', 'speed', 'steplen', 'strideT', 'bmi', 'percentStanceSS' , 'cadence'};

% Read Input Data
original_data = readtable('alldata.csv');

% Read Only One View 
data = original_data(char(original_data.side) == 'R', :);

% Read (Feature + Target) Values
data = data(:, Vars);

% Create Arrays from Tables
whole_sim_class = table2array(data);

% Clean NAN Inputs
index_nan = isnan(whole_sim_class);
test = not(logical(sum(index_nan, 2)));
whole_sim_class = whole_sim_class(test, :);

% Shuffle Data
rng(1);
[m, n] = size(whole_sim_class);
rand_ind = randperm(m);
whole_sim_class = whole_sim_class(rand_ind, :);

% Choose What Fraction of Data is Needed
fraction = 1;
new_m = floor(size(whole_sim_class, 1) / fraction);
new_n = size(whole_sim_class, 2);
whole_sim_class = whole_sim_class(1:new_m, :);

% Normalize Data
whole_sim_class  = whole_sim_class ./ max(whole_sim_class);

% Specify Bin Range
range = (0 : 0.1 : 1);

% Bucketize Target Value
target = whole_sim_class(:, end);
Y = discretize(target, range);
whole_sim_class(:, end) = Y;

% Divide Data With Ratio: [80, 20]
train_whole_sim_class = array2table(whole_sim_class(1 : 0.8*new_m, :));
test_whole_sim_class = array2table(whole_sim_class(0.8*new_m+1 : new_m, :));

% Add Back Column Titles
train_whole_sim_class.Properties.VariableNames = Vars;
test_whole_sim_class.Properties.VariableNames = Vars;

% Write Output Files
writetable(train_whole_sim_class, 'train_whole_sim_class.csv');
writetable(test_whole_sim_class, 'test_whole_sim_class.csv');


