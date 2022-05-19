

%% Script to generate Figure 3
%% from the ICLR paper
clear all
close all

runs = 10;
epochs = 1000;

% folder1 = 'Experiment1';
% folder2 = ['RNN', 'LSTM', 'GRU'];
% folder3RNN = ['16','64'];
% folder3GRU = ['8','32'];
runsstrings = {'Run1','Run2','Run3','Run4','Run5',...
  'Run6', 'Run7','Run8','Run9','Run10'};


% RNN
rmseRNN16T = zeros(runs,epochs);
rmseRNN64T = zeros(runs,epochs);
rmseRNN16V = zeros(runs,epochs);
rmseRNN64V = zeros(runs,epochs);
for i = 1:runs
  filename16 = ['Experiment1/RNN/16/',runsstrings{i},'/training_history.dat'];
  filename64 = ['Experiment1/RNN/64/',runsstrings{i},'/training_history.dat'];
%   filename16 = 'test.dat';
  mseRNN16 = load(filename16);
  mseRNN64 = load(filename64);
  
  rmseRNN16T(i,:) = sqrt(mseRNN16(:,1))';
  rmseRNN16V(i,:) = sqrt(mseRNN16(:,2))';
  
  rmseRNN64T(i,:) = sqrt(mseRNN64(:,1))';
  rmseRNN64V(i,:) = sqrt(mseRNN64(:,2))';
end

% LSTM
rmseLSTM8T = zeros(runs,epochs);
rmseLSTM32T = zeros(runs,epochs);
rmseLSTM8V = zeros(runs,epochs);
rmseLSTM32V = zeros(runs,epochs);
for i = 1:runs
  filename8 = ['Experiment1/LSTM/8/',runsstrings{i},'/training_history.dat'];
  filename32 = ['Experiment1/LSTM/32/',runsstrings{i},'/training_history.dat'];
  
  mseLSTM8 = load(filename8);
  mseLSTM32 = load(filename32);
  
  rmseLSTM8T(i,:) = sqrt(mseLSTM8(:,1))';
  rmseLSTM8V(i,:) = sqrt(mseLSTM8(:,2))';
  
  rmseLSTM32T(i,:) = sqrt(mseLSTM32(:,1))';
  rmseLSTM32V(i,:) = sqrt(mseLSTM32(:,2))';
end

% GRU
rmseGRU8T = zeros(runs,epochs);
rmseGRU32T = zeros(runs,epochs);
rmseGRU8V = zeros(runs,epochs);
rmseGRU32V = zeros(runs,epochs);
for i = 1:runs
  filename8 = ['Experiment1/GRU/8/',runsstrings{i},'/training_history.dat'];
  filename32 = ['Experiment1/GRU/32/',runsstrings{i},'/training_history.dat'];
  
  mseGRU8 = load(filename8);
  mseGRU32 = load(filename32);
  
  rmseGRU8T(i,:) = sqrt(mseGRU8(:,1))';
  rmseGRU8V(i,:) = sqrt(mseGRU8(:,2))';
  
  rmseGRU32T(i,:) = sqrt(mseGRU32(:,1))';
  rmseGRU32V(i,:) = sqrt(mseGRU32(:,2))';
end

%% mean of each column
rmseRNN16T_avg =  mean(rmseRNN16T);
rmseRNN16V_avg  = mean(rmseRNN16V);
rmseRNN64T_avg  = mean(rmseRNN64T);
rmseRNN64V_avg  = mean(rmseRNN64V);

rmseLSTM8T_avg  = mean(rmseLSTM8T);
rmseLSTM8V_avg  = mean(rmseLSTM8V);
rmseLSTM32T_avg = mean(rmseLSTM32T); 
rmseLSTM32V_avg = mean(rmseLSTM32V);

rmseGRU8T_avg  = mean(rmseGRU8T); 
rmseGRU8V_avg = mean(rmseGRU8V);
rmseGRU32T_avg = mean(rmseGRU32T);
rmseGRU32V_avg = mean(rmseGRU32V);

%% Plots
leg3a = {'RNN 16 training','RNN 16 validation',...
  'LSTM 8 training','LSTM 8 validation',...
  'GRU 8 training','GRU 8 validation'};
leg3b = {'RNN 64 training','RNN 64 validation',...
  'LSTM 32 training','LSTM 32 validation',...
  'GRU 32 training','GRU 32 validation'};


figure(3)  % Figure 3 left
semilogx(1:epochs,rmseRNN16T_avg,'r','LineWidth',2);
hold on
semilogx(1:epochs,rmseRNN16V_avg,'r-.','LineWidth',2);
hold on
semilogx(1:epochs,rmseLSTM8T_avg,'g','LineWidth',2);
hold on
semilogx(1:epochs,rmseLSTM8V_avg,'g-.','LineWidth',2);
hold on
semilogx(1:epochs,rmseGRU8T_avg,'b','LineWidth',2);
hold on
semilogx(1:epochs,rmseGRU8V_avg,'b-.','LineWidth',2);
hold on
legend(leg3a);
set(gca, 'FontSize', 14);
xlabel('Epochs');
ylabel('RMSE');

figure(4) % Figure 3 right
semilogx(1:epochs,rmseRNN64T_avg,'r','LineWidth',2);
hold on
semilogx(1:epochs,rmseRNN64V_avg,'r-.','LineWidth',2);
hold on
semilogx(1:epochs,rmseLSTM32T_avg,'g','LineWidth',2);
hold on
semilogx(1:epochs,rmseLSTM32V_avg,'g-.','LineWidth',2);
hold on
semilogx(1:epochs,rmseGRU32T_avg,'b','LineWidth',2);
hold on
semilogx(1:epochs,rmseGRU32V_avg,'b-.','LineWidth',2);
hold on
legend(leg3b);
set(gca, 'FontSize', 14);
xlabel('Epochs');
ylabel('RMSE');