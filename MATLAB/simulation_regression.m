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
whole_sim_reg = table2array(data);

% Clean NAN Inputs
index_nan = isnan(whole_sim_reg);
test = not(logical(sum(index_nan, 2)));
whole_sim_reg = whole_sim_reg(test, :);

% Shuffle Data
rng(1);
[m, n] = size(whole_sim_reg);
rand_ind = randperm(m);
whole_sim_reg = whole_sim_reg(rand_ind, :);

% Choose What Fraction of Data is Needed
fraction = 1;
new_m = floor(size(whole_sim_reg, 1) / fraction);
new_n = size(whole_sim_reg, 2);
whole_sim_reg = whole_sim_reg(1:new_m, :);

% Normalize Data
whole_sim_reg  = whole_sim_reg ./ max(whole_sim_reg);


% Divide Data With Ratio: [80, 15]
train_whole_sim_reg = array2table(whole_sim_reg(1 : 0.8*new_m, :));
test_whole_sim_reg = array2table(whole_sim_reg(0.8*new_m+1 : new_m, :));

% Add Back Column Titles
train_whole_sim_reg.Properties.VariableNames = Vars;
test_whole_sim_reg.Properties.VariableNames = Vars;

% Write Output Files
writetable(train_whole_sim_reg, 'train_whole_sim_reg.csv');
writetable(test_whole_sim_reg, 'test_whole_sim_reg.csv');


