clear variables; clc;

% Features
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

gmfcs_whole_sim_class = [whole_sim_class target];

% Divide Data With Ratio: [80, 20]
gmfcs_train_whole_sim_class = gmfcs_whole_sim_class(1 : 0.8*new_m, :);
gmfcs_test_whole_sim_class = gmfcs_whole_sim_class(0.8*new_m+1 : new_m, :);


% Write Output Files
csvwrite('gmfcs_whole_sim_class.csv', gmfcs_whole_sim_class);

% Write Output Files for Train and Test

csvwrite('gmfcs_train_whole_sim_class.csv', gmfcs_train_whole_sim_class);
csvwrite('gmfcs_test_whole_sim_class.csv', gmfcs_test_whole_sim_class);

