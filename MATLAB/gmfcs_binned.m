clear variables; clc;

% Features + Target
Vars = [4:71, 76:240];

% Read Input Data
original_data = readtable('alldata.csv');

% Read Only One View 
data = original_data(char(original_data.side) == 'R', :);

% Read Feature and Target Values
target = data(:, 'gmfcs');
data = data(:, Vars);

% Create Arrays from Tables
whole_sim_class = table2array(data);
target = table2array(target);

% Shuffle Data
rng(1);
[m, n] = size(whole_sim_class);
rand_ind = randperm(m);
whole_sim_class = whole_sim_class(rand_ind, :);
target = target(rand_ind, :);

% Choose What Fraction of Data is Needed
fraction = 1;
new_m = floor(size(whole_sim_class, 1) / fraction);
new_n = size(whole_sim_class, 2);
whole_sim_class = whole_sim_class(1:new_m, :);
target = target(1:new_m);

% Normalize Data
whole_sim_class  = whole_sim_class ./ max(whole_sim_class);

% Specify Bin Range
range = [0, 0.5, 2.5, 6];

% Bucketize Target Value
target = discretize(target, range);

whole_sim_class = [whole_sim_class target];

% Divide Data With Ratio: [100, 0]
train_whole_sim_class = array2table(whole_sim_class(1 : 1*new_m, :));

% Write Output Files
writetable(train_whole_sim_class, 'gmfcs_2groups_train_whole_sim_class.csv');

