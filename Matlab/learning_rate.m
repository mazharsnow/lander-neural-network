% lander_nn_matlab.m
% End-to-end: load data, train NN, plot RMSE & learning parameter (mu)

%% 1. Load data and build X, T
% Make sure this script is in the same folder as lander_train.csv
% or cd into that folder before running:
% cd('C:\Users\opc\Videos\Assignment Code')

lander_train = readmatrix('lander_train.csv');   % [N x 4] scaled data
% Columns: [x_target, y_target, vel_y, vel_x]

% Inputs: x_target, y_target  (N x 2)
X = lander_train(:, 1:2);

% Targets: vel_x, vel_y  (N x 2)  <-- note the swapped order
T = [lander_train(:, 4), ...      % vel_x
     lander_train(:, 3)];         % vel_y

% For MATLAB NN toolbox, inputs & targets should be [features x samples]
x = X';   % size: [2 x N]
t = T';   % size: [2 x N]

%% 2. Create and configure the network
% Training function: Levenberg-Marquardt backpropagation
trainFcn = 'trainlm';

% Architecture: 2-6-2 (6 hidden neurons, as per assignment)
hiddenLayerSize = 6;
net = fitnet(hiddenLayerSize, trainFcn);

% Limit training to ~100 epochs (assignment spec)
net.trainParam.epochs = 100;

% Data division (70% train, 15% val, 15% test)
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio   = 15/100;
net.divideParam.testRatio  = 15/100;

%% 3. Train the network
[net, tr] = train(net, x, t);

%% 4. Test the network on all data (for overall performance number)
y = net(x);
e = gsubtract(t, y);
performance = perform(net, t, y);   % overall MSE on all data
overall_RMSE = sqrt(performance);   % overall RMSE
fprintf('Overall MSE:  %f\n', performance);
fprintf('Overall RMSE: %f\n', overall_RMSE);

%% 5. RMSE over epochs (Train / Validation / Test)

% MSE curves from training record
mse_train = tr.perf;   % training MSE per epoch
mse_val   = tr.vperf;  % validation MSE per epoch
mse_test  = tr.tperf;  % test MSE per epoch

% Convert to RMSE
rmse_train = sqrt(mse_train);
rmse_val   = sqrt(mse_val);
rmse_test  = sqrt(mse_test);

% Epoch index (starts at 0)
epochs = 0:(numel(rmse_train)-1);

% Plot RMSE curves
figure;
plot(epochs, rmse_train, '-o'); hold on;
plot(epochs, rmse_val,   '-x');
plot(epochs, rmse_test,  '-s');
xlabel('Epoch');
ylabel('RMSE');
title('RMSE over Epochs (Train / Validation / Test)');
legend('Train', 'Validation', 'Test', 'Location', 'best');
grid on;

%% 6. Learning parameter mu over epochs (trainlm specific)

% For trainlm, tr.mu stores the LM "learning parameter"
mu = tr.mu;

figure;
semilogy(epochs, mu, '-o');   % log scale is useful if mu changes a lot
xlabel('Epoch');
ylabel('\mu');
title('Levenberg-Marquardt Learning Parameter \mu over Epochs');
grid on;

%% 7. Optional: built-in training state plot

figure;
plottrainstate(tr);
title('Training State (gradient, \mu, validation checks)');
