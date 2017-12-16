clear variables; clc;

Vars = {'AnkleDorsi_meanStance', 'AnkleDorsi_maxSwing', 'FootProg_meanStance'...
    , 'hasKinetics', 'age', 'speed', 'steplen', 'strideT', 'bmi', 'percentStanceSS' , 'cadence'};

original_data = readtable('alldata.csv');
data = original_data(char(original_data.side) == 'R', :);
data = data(:, Vars);


whole_sim_class = table2array(data);
index_nan = isnan(whole_sim_class);
test = not(logical(sum(index_nan, 2)));
whole_sim_class = whole_sim_class(test, :);


rng(1);
[m, n] = size(whole_sim_class);
rand_ind = randperm(m);
whole_sim_class = whole_sim_class(rand_ind, :);

new_m = floor(size(whole_sim_class, 1) / 1);
new_n = size(whole_sim_class, 2);

whole_sim_class = whole_sim_class(1:new_m, :);

%normalize
whole_sim_class  = whole_sim_class ./ max(whole_sim_class);


range = (0 : 0.1 : 1);

CAD = whole_sim_class(:, end);

Y = discretize(CAD, range);

whole_sim_class(:, end) = Y;

% 80 20 %

train_whole_sim_class = array2table(whole_sim_class(1 : 0.8*new_m, :));
test_whole_sim_class = array2table(whole_sim_class(0.8*new_m+1 : new_m, :));

train_whole_sim_class.Properties.VariableNames = Vars;
test_whole_sim_class.Properties.VariableNames = Vars;

writetable(train_whole_sim_class, 'train_whole_sim_class.csv');
writetable(test_whole_sim_class, 'test_whole_sim_class.csv');


