

%% Script to generate Figure 8
clear all
close all

runs = 10;
epochs = 1000;

hunits = {'4', '8'};
runsstrings = {'Run1','Run2','Run3','Run4','Run5',...
  'Run6', 'Run7','Run8','Run9','Run10'};

rmseGRUT = zeros(runs,epochs,length(hunits));
rmseGRUV = zeros(runs,epochs,length(hunits));

% All GRU
for hu = 1:length(hunits)
  for i = 1:runs
    filename1 = ['Experiment3/Dataset1/',hunits{hu},'/',runsstrings{i},'/training_history.dat'];
    filename2 = ['Experiment3/Dataset2/',hunits{hu},'/',runsstrings{i},'/training_history.dat'];
    
    mseGRU1 = load(filename1);
    mseGRU2 = load(filename2);

    rmseGRU1T(i,:,hu) = sqrt(mseGRU1(:,1))';
    rmseGRU1V(i,:,hu) = sqrt(mseGRU1(:,2))';
    rmseGRU2T(i,:,hu) = sqrt(mseGRU2(:,1))';
    rmseGRU2V(i,:,hu) = sqrt(mseGRU2(:,2))';
  end
end


%% mean along the runs
rmseGRU1T_avg = zeros(epochs,length(hu));
rmseGRU1V_avg = zeros(epochs,length(hu));
rmseGRU2T_avg = zeros(epochs,length(hu));
rmseGRU2V_avg = zeros(epochs,length(hu));
for hu = 1:length(hunits)
  rmseGRU1T_avg(:,hu) = mean(rmseGRU1T(:,:,hu))';
  rmseGRU1V_avg(:,hu) = mean(rmseGRU1V(:,:,hu))';
  rmseGRU2T_avg(:,hu) = mean(rmseGRU2T(:,:,hu))';
  rmseGRU2V_avg(:,hu) = mean(rmseGRU2V(:,:,hu))';
end


%% Plots
leg7 = {'Dataset 1 - 4 - Training','Dataset 1 - 4 - Validation',...
  'Dataset 2 - 4 - Training','Dataset 2 - 4 - Validation',...
  'Dataset 1 - 8 - Training','Dataset 1 - 8 - Validation',...
  'Dataset 2 - 8 - Training','Dataset 2 - 8 - Validation'};

ft = {'k','r','g','b','m'};
fv = {'k-.','r-.','g-.','b-.','m-.'};


xe = linspace(1,epochs,epochs);
idxf = 0;
figure(7) % Figure 7
for hu = 1:length(hunits)
%   set(gca, 'XScale', 'log');
%   hold on
idxf = idxf + 1;
  semilogx(xe,rmseGRU1T_avg(:,hu)',ft{idxf},'LineWidth',2);
  hold on
  semilogx(xe,rmseGRU1V_avg(:,hu),fv{idxf},'LineWidth',2);
  hold on
  semilogx(xe,rmseGRU2T_avg(:,hu)',ft{idxf+2},'LineWidth',2);
  hold on
  semilogx(xe,rmseGRU2V_avg(:,hu),fv{idxf+2},'LineWidth',2);
end
figure(7); hold on; legend(leg7);
set(gca, 'FontSize', 14);
xlabel('Epochs');
ylabel('RMSE');

