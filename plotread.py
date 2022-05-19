import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def getdata(data_path, ext):
    files_list = []
    files = os.listdir(data_path)
    for file in files:
        if file.endswith(ext):
            files_list.append(file)

    return files_list

def splitdata(data_list):
    train = []
    valid = []
    test = []
    saved_list = []
    valid_set = [37, 33, 21, 19, 17, 15, 7, 5, 2, 1]
    test_set = [39, 36, 35, 34, 32, 27, 25, 24, 18, 11]

    for file in data_list:
        saved_list.append(file)
    for i in range(len(test_set)):
        test.append(data_list.pop(test_set[i]-1))
    for i in range(len(test_set)):
        valid.append(saved_list.pop(valid_set[i]-1))
    train = [x for x in data_list if x in saved_list]

    return train, valid, test


def readdata(data_path, file_list, cols, nin, nout):
    datax = []
    datay = []
    for file in file_list:
        df = pd.read_csv(data_path + file,
            sep = "\s+", #separator whitespace
            names=cols)
        df = df.set_index('time')
        x = df.drop(columns = nout[1:5])
        y = df.drop(columns = nin[1:5])
        datax.append(x)
        datay.append(y)

    return datax, datay

def plotdata(sequences, dtype, var, folder, path):
    
    i=0
    for seq in sequences:
        seq.plot(kind='line')
        plt.legend(loc='upper left')
        Path(path + folder + dtype).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + folder + dtype + var + str(i) + '_plots.pdf')
        plt.close()

        i = i + 1

