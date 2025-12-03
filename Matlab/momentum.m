% momentum.m
% Compare different momentum values (mc) using traingdm
% and plot RMSE vs epoch for each.

%% 1. Load data and build X, T
% Make sure you are in the folder with lander_train.csv, or do:
% cd('C:\Users\opc\Videos\Assignment Code')

lander_train = readmatrix('lander_train.csv');   % [N x 4] scaled data
% Columns: [x_target, y_target, vel_y, vel_x]

% Inputs: x_target, y_target  (N x 2)
X = lander_train(:, 1:2);

% Targets: vel_x, vel_y  (N x 2)
T = [lander_train(:, 4), ...      % vel_x
     lander_train(:, 3)];         % vel_y

% For MATLAB NN toolbox, use [features x samples]
x = X';   % [2 x N]
t = T';   % [2 x N]

%% 2. Define training function and hyperparameters

trainFcn = 'traingdm';     % Gradient descent with momentum
hiddenLayerSize = 6;       % same architecture: 2-6-2
maxEpochs = 100;           % ~100 epochs

% Learning rate (for traingdm)
base_lr = 0.01;

% Momentum values to compare
mc_values = [0.0, 0.5, 0.9];

% Colors/markers for plotting
plotStyles = {'-o', '-x', '-s'};

%% 3. Loop over different momentum constants and train

figure; hold on;
legends = {};

for k = 1:numel(mc_values)
    mc = mc_values(k);
    
    % Create a network with traingdm
    net = fitnet(hiddenLayerSize, trainFcn);
    
    % Set training parameters
    net.trainParam.epochs = maxEpochs;
    net.trainParam.lr     = base_lr;   % learning rate
    net.trainParam.mc     = mc;        % momentum constant
    
    % Data division
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio   = 15/100;
    net.divideParam.testRatio  = 15/100;
    
    % Train the network
    [net, tr] = train(net, x, t);
    
    % Training MSE per epoch -> RMSE
    mse_train = tr.perf;
    rmse_train = sqrt(mse_train);
    
    % Epoch index (0-based)
    epochs = 0:(numel(rmse_train)-1);
    
    % Plot RMSE curve for this momentum value
    plot(epochs, rmse_train, plotStyles{k}, 'LineWidth', 1.2);
    legends{end+1} = sprintf('mc = %.1f', mc);
    
    % Print final RMSE to command window
    fprintf('Momentum mc = %.1f, final train RMSE = %f\n', mc, rmse_train(end));
end

xlabel('Epoch');
ylabel('RMSE');
title('Effect of Momentum (mc) on RMSE over Epochs (traingdm)');
legend(legends, 'Location', 'best');
grid on;
hold off;
