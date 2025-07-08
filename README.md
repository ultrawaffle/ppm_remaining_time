# Remaining Time Prediction: A Case Study

## About
This repository contains the evaluation scripts as described in the manuscript <i>Remaining Time Prediction: A Case Study</i> submitted to the ML4PM Workshop, co-located with the ICPM 2025.

## Data
The used data the outbound warehouse process can be found online: HIER BITE FIGSHARE LINK EINFÜGEN

This data set consists of:
```
- outbound_2024_ext_obfuscated.csv (complete event log without any pre-processing)
- outbound_2024_obfuscated.csv (event log after pre-processing step 1: without outliers)
- outbound_2024_obfuscated.csv (event log after pre-processing step 2: selected time frame - this log was used for all experiments)
```

Notice that all event logs have been anonymized due to privacy reasons. In case of any questions, please reach out to the authors.


### Built with
* ![platform](https://img.shields.io/badge/MacOS--9cf?logo=Apple&style=social)
* ![python](https://img.shields.io/badge/python-black?logo=python&label=3.9)

### Setup

Use the ```environment.yml``` to create a new conda environment with ```conda env create --file environment.yml```.

Copy the data folder over, which contains the prepared and obfuscated log. 
In particular, make sure to copy the logs from the link above into the ```/data/logs``` folder.   
Scripts may only run on Apple Silicon as the PyTorch implementations were altered.

<!-- ## Prepare Data

Running the file ```prepare_datasets.py [dataset] [input dataset location] [prefixes location]``` performs the following steps:
- Split dataset into training, validation, and test sets
- Calculate timestamp related features
- Prepare prefix data for PGT-Net and save to disk
- Prepare prefix data for DA-LSTM and save to disk

The input parameters are as follows:

- dataset: Name of the event log, excluding file extension
- input dataset location: General information like the data split are stored into this folder
- prefixes location: Folder to which the prefixes for training are saved.


Example:
```python prepare_datasets.py bpic2015_1 data/datasets/ data/preprocessed/``` -->

## Run experiments

### Prepare dataset
Running the file ```prepare_datasets.py [dataset] [input dataset location] [prefixes location]``` performs the following steps:
- Split dataset into training, validation, and test sets
- Calculate timestamp related features
- Prepare prefix data for PGT-Net and save to disk
- Prepare prefix data for DA-LSTM and save to disk

The input parameters are as follows:

- dataset: Name of the event log, excluding file extension
- input dataset location: General information like the data split are stored into this folder
- prefixes location: Folder to which the prefixes for training are saved.


Example:
```
python prepare_datasets.py outbound_2024_06_10_obfuscated data/logs data/preprocessed
```

### Model training
To start experiment, run the following command:
```python main.py [dataset] [model_type] [seed]```


- ```dataset: dataset name i.e., outbound```
- ```model_type: xgboostl1, lstm, pgtnet, sutran```
- ```seed: random seed```


Example:
```
python main.py outbound xgboostl1 1
```

## Project Organization

    ├── cml                                              <- In this folder, helper scripts are located
    ├── configs                                          <- This folder contains relevant configurations
    ├── CRTP_LSTM                                        <- This folder contains the code for the LSTM baseline
    ├── data                                             <- In this folder, make sure to store the event logs referenced above in a sub-folder '/logs' as well as an empty folder '/preprocessed'
    ├── graphgps                                         <- This folder contains the code for the PGT-Net baseline 
    ├── lstm                                             <- This folder contains helper scripts for the LSTM baseline
    ├── Preprocessing                                    <- This folder contains the scripts for pre-processing
    ├── SuTraN                                           <- This folder contains the code for the SuTraN baseline
    ├ const.py                                           <- constants
    ├ data_preprocessing.py                              <- main file for pre-processing
    ├ environment.yml                                    <- creating the environment
    ├ feature_calculation.py                             <- main file for feature calculation
    ├ main.py                                            <- main file for the whole project
    ├ pre_processing_data_exploration                    <- explorative notebook that can be used for pre-processing (equal to prepare_datasets.py) and for exploration of the event logs
    ├ prepare_datasets.py                                <- main file for event log preparation
    ├── README.md                                        <- The top-level README for users of this project.
    └── LICENSE  

### References

This project uses implementations of the models by the respective authors:
1. SuTraN / LSTM: https://github.com/BrechtWts/SuffixTransformerNetwork 
2. XGBoost / LSTM: https://github.com/RoiderJ/assessing_remaining_time_methods
3. PGTNet: https://github.com/keyvan-amiri/PGTNet
