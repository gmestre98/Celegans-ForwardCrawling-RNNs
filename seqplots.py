import pandas as pd
import matplotlib.pyplot as plt

##################################################################################################
##################################################################################################
# Experiment 1 - Obtains Plots for Figure 5 of the Article
##################################################################################################
##################################################################################################
mainfolder1 = "Experiment1/"
folder1_list = ["RNN/16/Run1/results_files/test", "LSTM/8/Run1/results_files/test", "GRU/8/Run1/results_files/test",
    "RNN/64/Run1/results_files/test", "LSTM/32/Run1/results_files/test", "GRU/32/Run1/results_files/test"]
sequences1 = ["3", "6"] #You can choose here more sequences from 0 to 9
end1 = "_data.dat"
cols11 = ["DB1", "DB1_Predicted"]
cols12 = ["LUAL", "LUAL_Predicted"]
cols21 = ["PVR", "PVR_Predicted"]
cols22 = ["VB1", "VB1_Predicted"]
color1 = ["red", "green"]
color2 = ["blue", "k"]
save1 = ["16RNN", "8LSTM", "8GRU", "64RNN", "32LSTM", "32GRU"]
head = ["time", "DB1", "LUAL", "PVR", "VB1", "DB1_Predicted", "LUAL_Predicted", "PVR_Predicted", "VB1_Predicted"]

i = 0
for f in folder1_list:
    for s in sequences1:
        df = pd.read_csv(mainfolder1 + f + s + end1,
            sep = "\s+",
            names=head)
        df = df.set_index('time')

        fig, ax = plt.subplots()
        df[cols11].plot(ax=ax, kind='line', color=color1)
        df[cols12].plot(ax=ax, kind='line', linestyle='dashed', marker='None', color=color1)
        ax.get_legend().remove()
        plt.xlabel("Time [sec]")
        plt.ylabel("Voltage")
        plt.savefig(mainfolder1 + 'Seq' + s + '_p1_' + save1[i] + '.pdf',
            bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots()
        df[cols21].plot(ax=ax, kind='line', color=color2)
        df[cols22].plot(ax=ax, kind='line', linestyle='dashed', marker='None', color=color2)
        ax.get_legend().remove()
        plt.xlabel("Time [sec]")
        plt.ylabel("Voltage")
        plt.savefig(mainfolder1 + 'Seq' + s + '_p2_' + save1[i] + '.pdf',
            bbox_inches='tight')
        plt.close()

    i = i + 1


##################################################################################################
##################################################################################################
# Experiment 2 - Obtains Plots for Figure 7 of the Article
##################################################################################################
##################################################################################################
mainfolder2 = "Experiment2/"
folder2_list = ["GRU/2/Run1/results_files/test", "GRU/4/Run1/results_files/test", "GRU/8/Run1/results_files/test", "GRU/16/Run1/results_files/test", "GRU/32/Run1/results_files/test", "GRU/64/Run1/results_files/test"]
sequences2 = ["2", "9"] #You can choose here more sequences from 0 to 9
end2 = "_data.dat"
save2 = ["2", "4", "8", "16", "32", "64"]

i = 0
for f in folder2_list:
    for s in sequences2:
        df = pd.read_csv(mainfolder2 + f + s + end2,
            sep = "\s+",
            names=head)
        df = df.set_index('time')

        fig, ax = plt.subplots()
        df[cols11].plot(ax=ax, kind='line', color=color1)
        df[cols12].plot(ax=ax, kind='line', linestyle='dashed', marker='None', color=color1)
        ax.get_legend().remove()
        plt.xlabel("Time [sec]")
        plt.ylabel("Voltage")
        plt.savefig(mainfolder2 + 'Seq' + s + '_p1_' + save2[i] + '.pdf',
            bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots()
        df[cols21].plot(ax=ax, kind='line', color=color2)
        df[cols22].plot(ax=ax, kind='line', linestyle='dashed', marker='None', color=color2)
        ax.get_legend().remove()
        plt.xlabel("Time [sec]")
        plt.ylabel("Voltage")
        plt.savefig(mainfolder2 + 'Seq' + s + '_p2_' + save2[i] + '.pdf',
            bbox_inches='tight')
        plt.close()

    i = i + 1


##################################################################################################
##################################################################################################
# Experiment 3 - Obtains Plots for Figure 9 of the Article
##################################################################################################
##################################################################################################
mainfolder3 = "Experiment3/"
folder3_list = ["Dataset1/4/Run1/results_files/test", "Dataset1/8/Run1/results_files/test", "Dataset2/4/Run1/results_files/test", "Dataset2/8/Run1/results_files/test"]
sequences3 = ["0", "4"] #You can choose here more sequences from 0 to 9
end3 = "_data.dat"
save3 = ["Data1_4", "Data1_8", "Data2_4", "Data2_8"]

i = 0
for f in folder3_list:
    for s in sequences3:
        df = pd.read_csv(mainfolder3 + f + s + end3,
            sep = "\s+",
            names=head)
        df = df.set_index('time')

        fig, ax = plt.subplots()
        df[cols11].plot(ax=ax, kind='line', color=color1)
        df[cols12].plot(ax=ax, kind='line', linestyle='dashed', marker='None', color=color1)
        ax.get_legend().remove()
        plt.xlabel("Time [sec]")
        plt.ylabel("Voltage")
        plt.savefig(mainfolder3 + 'Seq' + s + '_p1_' + save3[i] + '.pdf',
            bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots()
        df[cols21].plot(ax=ax, kind='line', color=color2)
        df[cols22].plot(ax=ax, kind='line', linestyle='dashed', marker='None', color=color2)
        ax.get_legend().remove()
        plt.xlabel("Time [sec]")
        plt.ylabel("Voltage")
        plt.savefig(mainfolder3 + 'Seq' + s + '_p2_' + save3[i] + '.pdf',
            bbox_inches='tight')
        plt.close()

    i = i + 1