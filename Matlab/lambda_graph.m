% lambda.m
% Compare different regularisation values (lambda)
% using net.performParam.regularization with trainlm
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

% For MATLAB NN toolbox: [features x samples]
x = X';   % [2 x N]
t = T';   % [2 x N]

%% 2. Training setup

trainFcn = 'trainlm';        % Levenberg-Marquardt
hiddenLayerSize = 6;         % 2-6-2
maxEpochs = 100;             % ~100 epochs (assignment)

% Different lambda (regularisation) values to test
lambda_values = [0.0, 0.1, 0.5];

% Plot styles
plotStyles = {'-o', '-x', '-s'};

%% 3. Loop over lambdas and train

figure; hold on;
legends = {};

for k = 1:numel(lambda_values)
    lambda = lambda_values(k);
    
    % Create network
    net = fitnet(hiddenLayerSize, trainFcn);
    
    % Set epochs
    net.trainParam.epochs = maxEpochs;
    
    % Set data division
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio   = 15/100;
    net.divideParam.testRatio  = 15/100;
    
    % Set regularisation (lambda)
    % 0   -> pure MSE
    % >0  -> mix of MSE and weight penalty
    net.performParam.regularization = lambda;
    
    % Train
    [net, tr] = train(net, x, t);
    
    % Training MSE per epoch -> RMSE
    mse_train = tr.perf;
    rmse_train = sqrt(mse_train);
    
    % Epoch index
    epochs = 0:(numel(rmse_train)-1);
    
    % Plot RMSE curve
    plot(epochs, rmse_train, plotStyles{k}, 'LineWidth', 1.2);
    legends{end+1} = sprintf('\\lambda = %.1f', lambda);
    
    % Print final RMSE to command window
    fprintf('lambda = %.1f, final train RMSE = %f\n', lambda, rmse_train(end));
end

xlabel('Epoch');
ylabel('RMSE');
title('Effect of Regularisation \lambda on RMSE (trainlm)');
legend(legends, 'Location', 'best');
grid on;
hold off;
