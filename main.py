import datetime
import pickle
import sys
from pathlib import Path
import numpy as np
import optuna
import shutil
#from graphgps.PGTNetutils import mean_cycle_norm_factor_provider
from graphgps.ResultHandler import CustomDataset
from lstm import objective as lstm_objective
from graphgps import objective as graphgps_objective
from cml import objective as cml_objective
import torch
from lstm import predict_test_set as lstm_predict_test_set
from graphgps import predict_test_set as graphgps_predict_test_set
import pandas as pd
import joblib
import xgboost as xgb
from cml import train
import os
import data_preprocessing
from graphgps import GTConverter
from lstm import prepare_dataset as lstm_prep
from CRTP_LSTM.TRAIN_EVAL_CRTP_LSTM_DA import train_eval as train_eval_lstm
from SuTraN.TRAIN_EVAL_SUTRAN_DA import train_eval as train_eval_sutran
#import multiprocessing as multiprocessing

#multiprocessing.set_start_method("fork")


# Add folder here!
base_path = '.'
device_name = 'mps'
device = torch.device(device_name)

def perform_optuna_study(input_data_location,
                         results_location,
                         model_type,
                         max_training_epochs,
                         cfg_file,
                         seed,
                         search_space=None):
    """
    Perform an Optuna study for the given model type.
    Parameters
    ----------
    input_data_location: str
        Folder where the input data is stored.
    results_location: str
        Folder where the results of the study will be stored.
    model_type: str
        Type of model to be used. Options are 'dalstm', 'graphgps', 'xgboost', 'xgboostl1'. 'xgboost' uses squared
        error function, 'xgboostl1' uses absolute error function.
    max_training_epochs: int
        Maximum number of training epochs for neural networks.
    cfg_file: str
        Path to the configuration file.
    seed: int
        Random eed for reproducibility.
    search_space: dict
        Search space for the hyperparameters. Key: str of parameter name, value: list of choices.

    Returns
    -------
    pd.DataFrame: Dataframe from trials of optuna study.
    """

    Path(results_location).mkdir(parents=True, exist_ok=True)
    p = str(Path(results_location).resolve())  # Get absolute path

    if Path(p + '/study.db').is_file():
        shutil.copyfile(p + '/study.db', p + '/study_safety_copy.db')

    # Choose objective and search space based on model type
    if model_type == 'dalstm':
        #device_name = f'cuda:0' if torch.cuda.is_available() else 'cpu'
        device_name = 'mps'
        device = torch.device(device_name)
        #device = device_name
        if search_space is None:
            search_space = {'n_neurons': [50, 100, 150],
                            'n_layers': [1, 2, 4]}
            #search_space = {'n_neurons': [100],
            #                'n_layers': [2]}
        objective = lstm_objective.Objective(cfg_file=cfg_file,
                                             input_data_location=input_data_location,
                                             output_location=results_location,
                                             device=device,
                                             seed=seed,
                                             max_training_epochs=max_training_epochs,
                                             search_space=search_space)

    elif model_type == 'pgtnet':
        if search_space is None:
            search_space = {'posenc_LapPE.dim_pe': [4, 8],
                            'posenc_RWSE.kernel.times_func_end': [8, 16]}
            # search_space = {'posenc_LapPE.dim_pe': [2, 4, 8],
            #     'posenc_RWSE.kernel.times_func_end': [4, 8, 16]}

        objective = graphgps_objective.Objective(cfg_file=cfg_file,
                                                 output_location=results_location,
                                                 input_location=input_data_location,
                                                 search_space=search_space,
                                                 seed=seed,
                                                 num_training_epochs=max_training_epochs)

    elif model_type == 'xgboost':
        if search_space is None:
            search_space = {'n_estimators': [100, 500],
                            'learning_rate': [0.1, 0.01],
                            'subsample': [0.3, 0.7],
                            'colsample_bytree': [0.3, 0.7],
                            'max_depth': [2, 4, 6]}
            # search_space = {'n_estimators': [100, 500],
            #     'learning_rate': [0.1, 0.01],
            #     'subsample': [0.3, 0.7],
            #     'colsample_bytree': [0.3, 0.7],
            #     'max_depth': [2]}

        objective = cml_objective.Objective(search_space=search_space,
                                            bucketing_method='single',
                                            encoding_method='combined',
                                            input_data_location=input_data_location,
                                            output_location=results_location,
                                            seed=seed,
                                            model_args={'n_jobs': -1,
                                                        'device': 'cuda',
                                                        'random_state': seed},
                                            cml_method=xgb.XGBRegressor)

    elif model_type == 'xgboostl1':
        if search_space is None:
            search_space = {'n_estimators': [100, 500],
                            'learning_rate': [0.1, 0.01],
                            'subsample': [0.3, 0.7],
                            'colsample_bytree': [0.3, 0.7],
                            'max_depth': [2, 4, 6]}
            # search_space = {'n_estimators': [500],
            #                 'learning_rate': [0.1],
            #                 'subsample': [0.3],
            #                 'colsample_bytree': [0.3],
            #                 'max_depth': [6]}

        objective = cml_objective.Objective(search_space=search_space,
                                            bucketing_method='single',
                                            encoding_method='combined',
                                            input_data_location=input_data_location,
                                            output_location=results_location,
                                            seed=seed,
                                            model_args={'n_jobs': -1,
                                                        'device': 'cuda',
                                                        'random_state': seed,
                                                        'objective': 'reg:absoluteerror',
                                                        'eval_metric':"mae"
                                                        # ,'early_stopping_rounds': 20
                                                        },
                                            cml_method=xgb.XGBRegressor)

    else:
        raise ValueError('Model type not supported')

    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(storage='sqlite:///' + p + '/study.db',
                                sampler=sampler,
                                pruner=None,
                                study_name='study',
                                direction='minimize',
                                load_if_exists=True,
                                directions=None)

    df = study.trials_dataframe()

    # Check how many trials were completed correctly
    completed_trials = list()
    for study in study.trials:
        if study.state == optuna.trial.TrialState.COMPLETE:
            completed_trials.append(study)
    num_completed_trials = len(completed_trials)

    # find number of hyperparameter combinations
    num_hp_trials = 1
    for v in search_space.values():
        num_hp_trials = num_hp_trials * len(v)

    # Do not train again if we have already the required number of trials
    if num_completed_trials >= num_hp_trials:
        df = df[df['state'] == 'COMPLETE']
        df.to_csv(results_location + '/trials.csv')
        print('Skipping tuning. All trials completed.')
        return df

    # Delete study before recreating to remove studies which failed. Otherwise, GridSearch won't be fully performed.
    optuna.delete_study(study_name='study', storage='sqlite:///' + p + '/study.db')

    sampler = optuna.samplers.GridSampler(search_space)
    # Recreate study and add trials to keep
    p = str(Path(results_location).resolve())
    study = optuna.create_study(storage='sqlite:///' + p + '/study.db',
                                sampler=sampler,  # TPESampler
                                pruner=None,
                                study_name='study',
                                direction='minimize',
                                load_if_exists=True,
                                directions=None)
    study.add_trials(completed_trials)

    # Start hyperparameter tuning
    study.optimize(objective, n_trials=int(num_hp_trials - num_completed_trials))

    df = study.trials_dataframe()
    df.to_csv(results_location + '/trials.csv', index=False)
    return df


def predict_test_set(best_trial,
                     model_type,
                     cfg_file,
                     input_data_location,
                     results_location,
                     seed):
    """
    Predicts the test set using the best trial found during hyperparameter tuning.

    Parameters
    ----------
    best_trial: pandas row
        Best trial found during hyperparameter tuning.
    model_type: str
        Model type to be used for prediction. Supported models are 'dalstm', 'graphgps', 'xgboost', 'xgboostl1'.
    cfg_file: str
        Configuration file for the model.
    input_data_location: str
           Location of the input data.
    results_location: str
        Location of the results.
    seed: int
        Random seed for reproducibility.

    Returns
    -------
    df: pandas DataFrame
        Dataframe containing predictions on the test set.
    """

    # Normalization factor
    train_df = pd.read_pickle(input_data_location + 'train.pkl')
    normalization_factor = train_df['remaining_time'].max()

    best_trial_num = best_trial['number']

    if model_type == 'dalstm':
        n_neurons = best_trial['params_n_neurons']
        n_layers = best_trial['params_n_layers']
        df = lstm_predict_test_set.predict(n_neurons=n_neurons,
                                      n_layers=n_layers,
                                      normalization_factor=normalization_factor,
                                      input_dataset_location=input_data_location + '/dalstm/data/',
                                      pretrained_model_filepath=results_location + '/trial_' + str(best_trial_num) + '/best_mae_ckpt.pt')

    elif model_type == 'pgtnet':
        model_name = os.listdir(results_location + '/trial_' + str(best_trial_num) + '/ckpt/')[0]
        df = graphgps_predict_test_set.predict(normalization_factor=normalization_factor,
                                               cfg_location=cfg_file,
                                               best_trial=best_trial,
                                               pretrained_model_filepath=results_location + '/trial_' + str(best_trial_num) + '/ckpt/' + model_name,
                                               data_location=input_data_location)

    elif model_type == 'xgboost':

        # Fit again, but also with validation set and only the best hyperparameters
        test_df = pd.read_pickle(input_data_location + '/test.pkl')

        train_df = train_df[train_df['remaining_time'] != 0.]
        test_df = test_df[test_df['remaining_time'] != 0.]

        config = joblib.load(input_data_location + '/config.pkl')

        train_df[config['timestamp_column']] = pd.to_datetime(train_df[config['timestamp_column']], format='mixed',
                                                              infer_datetime_format=True)
        train_df = train_df.sort_values(by=[config['case_id_column'], config['timestamp_column']])

        test_df[config['timestamp_column']] = pd.to_datetime(test_df[config['timestamp_column']], format='mixed',
                                                             infer_datetime_format=True)
        test_df = test_df.sort_values(by=[config['case_id_column'], config['timestamp_column']])

        encoding_args = {'case_id_col': config['case_id_column'],
                         'static_cat_cols': config['static_categorical_columns'],
                         'static_num_cols': config['static_numerical_columns'],
                         'dynamic_cat_cols': config['dynamic_categorical_columns'],
                         'dynamic_num_cols': config['dynamic_numerical_columns'],
                         'fillna': True}
        df = train.train(train_df,
                         test_df,
                         config=config,
                         bucketing_method='single',
                         random_state=seed,
                         encoding_methods=["static", "last", "agg"],
                         encoding_args=encoding_args,
                         cls_method=xgb.XGBRegressor,
                         cls_args={'n_jobs': -1,
                                   'random_state': seed,
                                   'n_estimators': best_trial['params_n_estimators'],
                                   'learning_rate': best_trial['params_learning_rate'],
                                   'subsample': best_trial['params_subsample'],
                                   'colsample_bytree': best_trial['params_colsample_bytree'],
                                   'max_depth': best_trial['params_max_depth']
                                   })

    elif model_type == 'xgboostl1':
        # Fit again, but also with validation set and only the best hyperparameters
        test_df = pd.read_pickle(input_data_location + '/test.pkl')

        train_df = train_df[train_df['remaining_time'] != 0.]
        test_df = test_df[test_df['remaining_time'] != 0.]

        config = joblib.load(input_data_location + '/config.pkl')

        train_df[config['timestamp_column']] = pd.to_datetime(train_df[config['timestamp_column']], format='mixed',
                                                              infer_datetime_format=True)
        train_df = train_df.sort_values(by=[config['case_id_column'], config['timestamp_column']])

        test_df[config['timestamp_column']] = pd.to_datetime(test_df[config['timestamp_column']], format='mixed',
                                                             infer_datetime_format=True)
        test_df = test_df.sort_values(by=[config['case_id_column'], config['timestamp_column']])

        encoding_args = {'case_id_col': config['case_id_column'],
                         'static_cat_cols': config['static_categorical_columns'],
                         'static_num_cols': config['static_numerical_columns'],
                         'dynamic_cat_cols': config['dynamic_categorical_columns'],
                         'dynamic_num_cols': config['dynamic_numerical_columns'],
                         'fillna': True}
        cls_args = {'n_jobs': -1,
                                   'random_state': seed,
                                   'n_estimators': best_trial['params_n_estimators'],
                                   'learning_rate': best_trial['params_learning_rate'],
                                   'subsample': best_trial['params_subsample'],
                                   'colsample_bytree': best_trial['params_colsample_bytree'],
                                   'max_depth': best_trial['params_max_depth'],
                                   'objective': 'reg:absoluteerror',
                                   'device': 'cuda'
                                   }
        print(cls_args)
        df = train.train(train_df,
                         test_df,
                         config=config,
                         bucketing_method='single',
                         random_state=seed,
                         encoding_methods=["static", "last", "agg"],
                         encoding_args=encoding_args,
                         cls_method=xgb.XGBRegressor,
                         cls_args=cls_args)

    return df

# def create_original_dataset_from_graph_df(graph_dataset_folder, results_location, prediction_dataframe):
    
#     # Normalization factor
#     train_df = pd.read_pickle(input_data_location + 'train.pkl')
#     normalization_factor = train_df['remaining_time'].max()

#     with open(os.path.join(graph_dataset_folder, "train.pickle"), "rb") as f:
#         train_dataset = CustomDataset(root=graph_dataset_folder, data_list=pickle.load(f))
#     with open(os.path.join(graph_dataset_folder, "val.pickle"), "rb") as f:
#         val_dataset = CustomDataset(root=graph_dataset_folder, data_list=pickle.load(f))
#     with open(os.path.join(graph_dataset_folder, "test.pickle"), "rb") as f:
#         test_dataset = CustomDataset(root=graph_dataset_folder, data_list=pickle.load(f))
#     combined_dataset = train_dataset + val_dataset + test_dataset
    
#     # use graph dataset to create a pandas dataframe
#     num_node_list = []
#     num_edge_list = []
#     real_cycle_time_list = []
#     cid_list = []
#     pl_list = []
#     for graph in combined_dataset:
#         num_nodes = graph.num_nodes
#         num_edges = graph.num_edges
#         real_cycle_time = graph.y.item()  # Assuming y is a single value tensor
#         cid = graph.cid
#         pl = graph.pl
#         num_node_list.append(num_nodes)
#         num_edge_list.append(num_edges)
#         real_cycle_time_list.append(real_cycle_time)
#         cid_list.append(cid)
#         pl_list.append(pl)
#     data = {"num_node": num_node_list, "num_edge": num_edge_list, "real_cycle_time": real_cycle_time_list,
#             "cid": cid_list, "pl": pl_list}
#     initial_dataframe = pd.DataFrame(data)
    
#     # match predictions and data used for training. 
#     final_result_dataframe = pd.DataFrame(columns=list(initial_dataframe.columns) + ['predicted_cycle_time'])
#     # Iterate through the initial DataFrame
#     for index, initial_row in initial_dataframe.iterrows():
#         num_node = initial_row['num_node']
#         num_edge = initial_row['num_edge']
#         real_cycle_time_init = initial_row['real_cycle_time']    
#         # Filter the prediction DataFrame based on "num_node" and "num_edge"
#         filtered_prediction = prediction_dataframe[(prediction_dataframe['num_node'] == num_node) & (prediction_dataframe['num_edge'] == num_edge)]    
#         if not filtered_prediction.empty:
#             # Find the closest match based on "real_cycle_time" within the filtered subset
#             closest_match = filtered_prediction.iloc[np.argmin(np.abs(filtered_prediction['real_cycle_time'] - real_cycle_time_init))] 
#             # Add the matched row to the final result DataFrame with "predicted_cycle_time"
# 	    # if you are using older version of pandas you might need to replace "_append" by "append" in the following lines.
#             match_data = initial_row._append(pd.Series([closest_match['predicted_cycle_time']], index=['predicted_cycle_time']))
#             final_result_dataframe = final_result_dataframe._append(match_data, ignore_index=True)
    
#     #normalization_factor, mean_cycle_time = mean_cycle_norm_factor_provider(dataset_name)
#     final_result_dataframe['MAE-days'] = (final_result_dataframe['real_cycle_time'] - final_result_dataframe['predicted_cycle_time']).abs() * normalization_factor
#     final_result_dataframe.to_csv(results_location, index=False)
#     return

if __name__ == '__main__':
    dataset = sys.argv[1]
    model_type = sys.argv[2]
    seed = int(sys.argv[3])
    #tuning = bool(sys.argv[4])
    tuning = True

    input_data_location = base_path + '/data/preprocessed/' + dataset + '/'
    print(base_path)
    raw_logs_location = base_path + '/data/logs/'

    if model_type != 'lstm' and model_type != 'sutran':
        print('Checking if data is prepared...')
        if not os.path.isfile(input_data_location + '/test.pkl'):
            print('preparing data...')
            data_preprocessing.prepare_data(dataset_name=dataset,
                                            dataset_location=raw_logs_location,
                                            output_location=base_path + '/data/preprocessed/')

        if model_type == 'dalstm' and not os.path.isfile(input_data_location +  '/dalstm/data/train.hdf5'):
            print('preparing data ', input_data_location +  '/dalstm/data/train.hdf5')
            lstm_prep.prepare_dataset(input_data_filepath=input_data_location + '/',
                                    output_data_filepath=input_data_location  + '/dalstm/data/')
        if model_type == 'pgtnet' and not os.path.isfile(input_data_location + '/graph_dataset/raw/train.pickle'):
            GTConverter.create_graph_dataset(input_dataset_location=input_data_location + '/',
                                            graph_dataset_path_raw=input_data_location + '/graph_dataset/raw/')

        results_location = base_path + '/results/' + dataset + '/' + model_type + '/seed_' + str(seed) + '/'
        model_type = model_type
        max_training_epochs = 200
        cfg_file = 'configs/' + model_type + '/general.yaml'

        if(tuning):
            print('Start optuna study...')
            # Hyperparameter tuning
            trials_df = perform_optuna_study(input_data_location=input_data_location,
                                            results_location=results_location + '/hp_tuning/',
                                            model_type=model_type,
                                            max_training_epochs=max_training_epochs,
                                            cfg_file=cfg_file,
                                            seed=seed)

            # Get model which performed best on validation set
            trials_df = trials_df[trials_df['state'] == 'COMPLETE']
            trials_df = trials_df[trials_df['value'] == trials_df['value'].min()]
            best_trial = trials_df.iloc[0]
            print('HP tuning completed...')
        else:
            #best_trial = pd.DataFrame({'params_n_layers': [2], 'params_n_neurons': [100]})
            best_trial = pd.DataFrame({'params_posenc_LapPE.dim_pe': [4], 'params_posenc_RWSE.kernel.times_func_end': [8]})
            # best_trial = pd.DataFrame({ 'number': [1],
            #                             'params_n_estimators': [500],
            #                            'params_learning_rate': [0.1],
            #                            'params_subsample': [0.3],
            #                            'params_colsample_bytree': [0.7],
            #                            'params_max_depth': [4],
            # })
                                    
        # Train best architecture several times
        #num_trainings = 10
        num_trainings = 1
        if model_type == 'pgtnet':
            search_space = {'posenc_LapPE.dim_pe': [int(best_trial['params_posenc_LapPE.dim_pe'])],
                            'posenc_RWSE.kernel.times_func_end': [int(best_trial['params_posenc_RWSE.kernel.times_func_end'])]}
        elif model_type == 'dalstm':
            search_space = {'n_neurons': [int(best_trial['params_n_neurons'])],
                            'n_layers': [int(best_trial['params_n_layers'])]}

        print('Start stability tests...')
        for i in range(num_trainings):
            results_location_ = results_location + '/stability_retraining/num_' + str(i) + '/'
            Path(results_location_).mkdir(parents=True, exist_ok=True)

            # Continue if model is already trained
            if os.path.exists(results_location_ + '/test_set_predictions.csv'):
                continue
            
            if model_type == 'dalstm' or model_type == 'pgtnet':
                trials_df = perform_optuna_study(input_data_location=input_data_location,
                                                results_location=results_location_,
                                                model_type=model_type,
                                                max_training_epochs=max_training_epochs,
                                                cfg_file=cfg_file,
                                                seed=i,
                                                search_space=search_space)
                print(f'{datetime.datetime.now().isoformat()}, Prediction Started')
                df = predict_test_set(best_trial=trials_df.iloc[0],
                                    model_type=model_type,
                                    cfg_file=cfg_file,
                                    input_data_location=input_data_location,
                                    results_location=results_location_,
                                    seed=i)
                # if model_type == 'graphgps':
                #     #add
                #     create_original_dataset_from_graph_df(input_data_location, results_location_, df)

            elif model_type == 'xgboost' or model_type == 'xgboostl1':
                print(f'{datetime.datetime.now().isoformat()}, Prediction Started')
                df = predict_test_set(best_trial=best_trial,
                                    model_type=model_type,
                                    cfg_file=cfg_file,
                                    input_data_location=input_data_location,
                                    results_location=results_location_,
                                    seed=i)
            print(f'{datetime.datetime.now().isoformat()}, Prediction Done')
            df.to_csv(results_location_ + '/test_set_predictions.csv', index=False)
            print('MAE: (min.) ', (df['labels'] - df['preds']).abs().mean()//60)
            median = df['labels'].median()
            print('nMAE: ', ((df['labels'] - df['preds']).abs().mean()) / ((median - df['labels']).abs().mean()))
    else:
        if(model_type == 'lstm'):
            train_eval_lstm(dataset, input_data_location + 'sutran/', base_path + '/results/' + dataset)
        elif(model_type == 'sutran'):
            train_eval_sutran(dataset, input_data_location + 'sutran/', base_path + '/results/' + dataset)
        
    print(f'{datetime.datetime.now().isoformat()}, Completed everything.')
