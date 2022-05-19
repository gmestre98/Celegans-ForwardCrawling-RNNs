# Modelling Neuronal Behaviour with Time Series Regression: Recurrent Neural Networks on C.*Elegans* Data

This repository is the official implementation of Modelling Neuronal Behaviour with Time Series Regression: Recurrent Neural Networks on C.*Elegans* Data. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  The available code was developed using Python 3.9.5.
>📋  To run the code the libraries available in the "requirements.txt" file were used, with the referred version.
>📋  It is recommended to install them with the given command and with the same version of each library, although other versions might still work.

## Training

To train the model(s) in the paper, run this command:

```train
python train.py -model 'Type of Layer' -epochs 1000 -hidden_size "Recurrent Layer Size" -learning_rate 0.001 -optimizer 'adam' -datapath <path do the data/> -savepath <path where to save the results to/> -batch_size 32 -plots True/False
```
>📋 The input parameters for the training script are:
    * -model: The model to use from, ['LSTM', 'GRU', 'RNN']
    * -epochs: The number of epochs to train the model
    * -hidden_size: The size of the recurrent neural network layer
    * -learning_rate: The value for the learning rate
    * -optimizer: The optimizer to use from ['sgd', 'rmsprop', 'adam']
    * -datapath: The path to take the data from
    * -savepath: The path where to save the weights, model and training history
    * -batch_size: Size of each batch of data used to train the model
    * -plots: Whether to plot the inputs and outputs from the dataset or not
    * -early_stop: Whether to use early stopping or not
    * -patience: The amount of epochs to wait with no improvement before stopping the model training in case -early_stop is true
>📋 The appropriate hyperparameters are learning_rate = 0.001 and batch_size = 32 for the RNN and learning_rate = 0.05 and batch_size = 32 for the LSTM and GRU
>📋 The rest of the parameters used for training each model depend on the different experiments and settings tried. To check the appropriate settings for each one go to the Results section of this README
>📋 This script only trains the model creating a folder with the weights saved in each iteration where the model improved, a training history data file with training and validation losses, "training_history.dat" and a saved model file "model.h5"

## Evaluation

To evaluate any of the models used run:

```eval
python eval.py -model 'Type of Layer' -datapath <path do the data/> -savepath <path where the model was saved/> -batch_size 20 -plots_in True/False -plots_out True/False
```

>📋 The input parameters for the training script are:
    * -model: The model to use from, ['LSTM', 'GRU', 'RNN'].
    * -datapath: The path to take the data from.
    * -savepath: The path where to save the model was saved.
    * -batch_size: Size of each batch of data used to evaluate the model.
    * -plots_in: Whether to plot the inputs and outputs from the dataset or not.
    * -plots_out: Whether to plot the results or not.
>📋 An example for each evaluation execution of each model can be found on the results section of this README
>📋 This script only evaluates the model creating a folder with .dat files containing the results and also creating a folder with plots of the results if -plots_out is True.

## Pre-trained Models

For each setting tried in each experiment there is an available model, "model.h5" in the corresponding folder.

## Results

Our article has 3 experiments made regarding different model settings as can be seen discussed next:

There are also made available too scripts which can be called to generate the Figures from the article.

For Figures 4, 6 and 8 run:
```
python seqplots.py
```

### Experiment 1

Experiment 1 consisted on verifying the most appropriate type of recurrent layer to use. There were tried simple RNNs with 16 and 64 neurons for the recurrent layer, and LSTMs and GRUs with 8 and 32 for the recurrent layer. To replicate each of the experiments use the following training and/or evaluation commands, depending whether you want to do both training and evaluation or just one of them.

>📋 RNN - Size 16

```train
python train.py -model 'RNN' -hidden_size 16 -datapath "Dataset1/" -savepath "Experiment1/RNN/16/Runx/"
```
```eval
python eval.py -model 'RNN' -datapath "Dataset1/" -savepath "Experiment1/RNN/16/Runx/"
```

>📋 LSTM - Size 8

```train
python train.py -model 'LSTM' -hidden_size 8 -datapath "Dataset1/" -savepath "Experiment1/LSTM/8/Runx/"
```
```eval
python eval.py -model 'LSTM' -datapath "Dataset1/" -savepath "Experiment1/LSTM/8/Runx/"
```

>📋 GRU - Size 8

```train
python train.py -model 'GRU' -hidden_size 8 -datapath "Dataset1/" -savepath "Experiment1/GRU/8/Runx/"
```
```eval
python eval.py -model 'GRU' -datapath "Dataset1/" -savepath "Experiment1/GRU/8/Runx/"
```

>📋 RNN - Size 64

```train
python train.py -model 'RNN' -hidden_size 64 -datapath "Dataset1/" -savepath "Experiment1/RNN/64/Runx/"
```
```eval
python eval.py -model 'RNN' -datapath "Dataset1/" -savepath "Experiment1/RNN/64/Runx/"
```

>📋 LSTM - Size 32

```train
python train.py -model 'LSTM' -hidden_size 32 -datapath "Dataset1/" -savepath "Experiment1/LSTM/32/Runx/"
```
```eval
python eval.py -model 'LSTM' -datapath "Dataset1/" -savepath "Experiment1/LSTM/32/Runx/"
```

>📋 GRU - Size 32

```train
python train.py -model 'GRU' -hidden_size 32 -datapath "Dataset1/" -savepath "Experiment1/GRU/32/Runx/"
```
```eval
python eval.py -model 'GRU' -datapath "Dataset1/" -savepath "Experiment1/GRU/32/Runx/"
```

>📋 The results obtained are summarized in the following table in which the values refer to the Root Mean Squared Error:

| Set / Model      |  RNN - 16  | LSTM - 8   |  GRU - 8   |  RNN - 64  | LSTM - 32  |  GRU - 32  | 
| ---------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Training Set     | 1.0444e-01 | 9.9371e-03 | 7.7912e-03 | 6.2831e-02 | 7.6058e-03 | 6.7129e-03 |
| Validation Set   | 1.2986e-01 | 3.6109e-02 | 3.4760e-02 | 8.8764e-02 | 3.6437e-02 | 3.5098e-02 |
| Test Set         | 1.3406e-01 | 1.4904e-02 | 1.0005e-02 | 9.5476e-02 | 1.5230e-02 | 1.2384e-02 |

### Experiment 2

Experiment 2 consisted on verifying the most appropriate size for the recurrent layer to use. 6 different sizes for the recurrent layer, 2, 4, 8, 16, 32 and 64. To replicate each of the experiments use the following training and/or evaluation commands, depending whether you want to do both training and evaluation or just one of them.


>📋 Size 2

```train
python train.py -model 'GRU' -hidden_size 2 -datapath "Dataset1/" -savepath "Experiment2/GRU/2/Runx/"
```
```eval
python eval.py -model 'GRU' -datapath "Dataset1_0.5/" -savepath "Experiment2/GRU/2/Runx/"
```

>📋 Size 4

```train
python train.py -model 'GRU' -hidden_size 4 -datapath "Dataset1/" -savepath "Experiment2/GRU/4/Runx/"
```
```eval
python eval.py -model 'GRU' -datapath "Dataset1_0.5/" -savepath "Experiment2/GRU/4/Runx/"
```

>📋 Size 8

```train
python train.py -model 'GRU' -hidden_size 8 -datapath "Dataset1/" -savepath "Experiment2/GRU/8/Runx/"
```
```eval
python eval.py -model 'GRU' -datapath "Dataset1_0.5/" -savepath "Experiment2/GRU/8/Runx/"
```

>📋 Size 16

```train
python train.py -model 'GRU' -hidden_size 16 -datapath "Dataset1/" -savepath "Experiment2/GRU/16/Runx/"
```
```eval
python eval.py -model 'GRU' -datapath "Dataset1_0.5/" -savepath "Experiment2/GRU/16/Runx/"
```

>📋 Size 32

```train
python train.py -model 'GRU' -hidden_size 32 -datapath "Dataset1/" -savepath "Experiment2/GRU/32/Runx/"
```
```eval
python eval.py -model 'GRU' -datapath "Dataset1_0.5/" -savepath "Experiment2/GRU/32/Runx/"
```

>📋 Size 64

```train
python train.py -model 'GRU' -hidden_size 64 -datapath "Dataset1/" -savepath "Experiment2/GRU/64/Runx/"
```
```eval
python eval.py -model 'GRU' -datapath "Dataset1_0.5/" -savepath "Experiment2/GRU/64/Runx/"
```


>📋 The results obtained are summarized in the following table in which the values refer to the Root Mean Squared Error:

| Set / Model      |  2 Units   |  4 Units   |  8 Units   |  16 Units  |  32 Units  |  64 Units  | 
| ---------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Training Set     | 4.2999e-02 | 1.0525e-02 | 7.7912e-03 | 6.7886e-03 | 6.7128e-03 | 5.8068e-03 |
| Validation Set   | 5.3732e-02 | 3.4800e-02 | 3.4760e-02 | 3.4961e-02 | 3.5098e-02 | 3.6230e-02 |
| Test Set         | 6.0047e-02 | 1.1771e-02 | 1.0005e-02 | 9.3625e-03 | 1.2384e-02 | 1.6841e-02 |

### Experiment 3

Experiment 3 consisted on testing the ability of the models chosen in previous experiments to handle data with different timesteps, having more data points. There were tested 2 datasets and 2 different sizes for the recurrent layer, 4, 8. To replicate each of the experiments use the following training and/or evaluation commands, depending whether you want to do both training and evaluation or just one of them.


>📋 Dataset 1 - 4 Units

```train
python train.py -model 'GRU' -hidden_size 4 -datapath "Dataset1/" -savepath "Experiment3/Dataset1/4/Runx/"
```
```eval
python eval.py -model 'GRU' -datapath "Dataset1/" -savepath "Experiment3/Dataset1/4/Runx/"
```

>📋 Dataset 2 - 4 Units

```train
python train.py -model 'GRU' -hidden_size 4 -datapath "Dataset2/" -savepath "Experiment3/Dataset2/4/Runx/"
```
```eval
python eval.py -model 'GRU' -datapath "Dataset2/" -savepath "Experiment3/Dataset2/4/Runx/"
```

>📋 Dataset 1 - 8 Units

```train
python train.py -model 'GRU' -hidden_size 8 -datapath "Dataset1/" -savepath "Experiment3/Dataset1/8/Runx/"
```
```eval
python eval.py -model 'GRU' -datapath "Dataset1/" -savepath "Experiment3/Dataset1/8/Runx/"
```

>📋 Dataset 2 - 8 Units

```train
python train.py -model 'GRU' -hidden_size 8 -datapath "Dataset2/" -savepath "Experiment3/Dataset2/8/Runx/"
```
```eval
python eval.py -model 'GRU' -datapath "Dataset2/" -savepath "Experiment3/Dataset2/8/Runx/"
```

>📋 The results obtained are summarized in the following table in which the values refer to the Root Mean Squared Error:

| Set / Model      | Dataset1 - 4 | Dataset2 - 4 | Dataset1 - 8 | Dataset2 - 8 |
| ---------------- | ------------ | ------------ | ------------ | ------------ |
| Training Set     |  1.0525e-02  |  1.0878e-02  |  7.7912e-03  |  8.1769e-03  |
| Validation Set   |  3.4800e-02  |  3.4554e-02  |  3.4760e-02  |  3.5440e-02  |
| Test Set         |  1.1771e-02  |  1.2107e-02  |  1.0005e-02  |  1.0434e-02  |


## Contributing

>📋  The License is available in this repository.