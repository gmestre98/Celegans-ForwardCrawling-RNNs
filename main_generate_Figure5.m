

%% Script to generate Figure 5
%% from the ICLR paper
clear all
close all

runs = 10;
epochs = 1000;

hunits = {'2', '4', '8', '16', '32','64'};
runsstrings = {'Run1','Run2','Run3','Run4','Run5',...
  'Run6', 'Run7','Run8','Run9','Run10'};

rmseGRUT = zeros(runs,epochs,length(hunits));
rmseGRUV = zeros(runs,epochs,length(hunits));

% All GRU
for hu = 1:length(hunits)
  for i = 1:runs
    filename = ['Experiment2/GRU/',hunits{hu},'/',runsstrings{i},'/training_history.dat'];
    
    mseGRU = load(filename);

    rmseGRUT(i,:,hu) = sqrt(mseGRU(:,1))';
    rmseGRUV(i,:,hu) = sqrt(mseGRU(:,2))';
  end
end


%% mean along the runs
rmseGRUT_avg = zeros(epochs,length(hu));
rmseGRUV_avg = zeros(epochs,length(hu));
for hu = 1:length(hunits)
  rmseGRUT_avg(:,hu) = mean(rmseGRUT(:,:,hu))';
  rmseGRUV_avg(:,hu) = mean(rmseGRUV(:,:,hu))';
end


%% Plots
leg5 = {'GRU 2 training','GRU 2 validation',...
  'GRU 4 training','GRU 4 validation',...
  'GRU 8 training','GRU 8 validation',...
  'GRU 16 training','GRU 16 validation',...
  'GRU 32 training','GRU 32 validation',...
  'GRU 64 training','GRU 64 validation'};

ft = {'k','r','g','y','b','m'};
fv = {'k-.','r-.','g-.','y-.','b-.','m-.'};

figure(5) % Figure 5
xe = linspace(1,epochs,epochs);
for hu = 1:length(hunits)
  t = rmseGRUT_avg(:,hu)';
  v = rmseGRUV_avg(:,hu)';
  set(gca, 'XScale', 'log');
  hold on
  semilogx(xe,t,ft{hu},'LineWidth',2);
  hold on
  semilogx(xe,v,fv{hu},'LineWidth',2);
end
figure(5); hold on; legend(leg5);
set(gca, 'FontSize', 14);
xlabel('Epochs');
ylabel('RMSE');

