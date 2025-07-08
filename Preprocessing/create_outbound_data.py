import pandas as pd 
import numpy as np
from Preprocessing.from_log_to_tensors import log_to_tensors
import os 
import torch


def construct_datasets(dataset_name, dataset_location, output_location):
    df = pd.read_csv(dataset_location + '/' + dataset_name + '.csv', index_col=False)
    output_directory = output_location + '/' + dataset_name + '/' + 'sutran' #log_name
    os.makedirs(output_directory, exist_ok=True)

    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format = 'mixed').dt.tz_convert('UTC')
    df = df.astype({'case:concept:name': str})
    
    categorical_casefeatures = ['CAT_ATTR_01',
                                   'CAT_ATTR_04',
                                   'CAT_ATTR_06',
                                   'CAT_ATTR_08',
                                   'CAT_ATTR_09',
                                   'CAT_ATTR_12',
                                   'CAT_ATTR_15'
                                   ]
    
    numeric_eventfeatures = ['concurrent_cases']
    categorical_eventfeatures = []
    numeric_casefeatures = ['NUM_ATTR_03', 
                            'NUM_ATTR_04'
                            ]
    case_id = 'case:concept:name'
    timestamp = 'time:timestamp'
    act_label = 'concept:name'
    print(df.info())

    start_date = None # "2024-08"
    end_date = None
    max_days = 50
    window_size = 15 ## 13 Events 
    log_name = dataset_name
    start_before_date = None
    test_len_share = 0.3
    val_len_share = 0.2
    mode = 'preferred'
    outcome = None
    result = log_to_tensors(df, 
                            log_name=log_name, 
                            start_date=start_date, 
                            start_before_date=start_before_date,
                            end_date=end_date, 
                            max_days=max_days, 
                            test_len_share=test_len_share, 
                            val_len_share=val_len_share, 
                            window_size=window_size, 
                            mode=mode,
                            case_id=case_id, 
                            act_label=act_label, 
                            timestamp=timestamp, 
                            cat_casefts=categorical_casefeatures, 
                            num_casefts=numeric_casefeatures, 
                            cat_eventfts=categorical_eventfeatures, 
                            num_eventfts=numeric_eventfeatures, 
                            outcome=outcome,
                            output_directory=output_directory)
    
    train_data, val_data, test_data = result

    # Create the log_name subfolder in the root directory of the repository
    # (Should already be created when having executed the `log_to_tensors()`
    # function.)


    # Save training tuples
    train_tensors_path = os.path.join(output_directory, 'train_tensordataset.pt')
    torch.save(train_data, train_tensors_path)

    # Save validation tuples
    val_tensors_path = os.path.join(output_directory, 'val_tensordataset.pt')
    torch.save(val_data, val_tensors_path)

    # Save test tuples
    test_tensors_path = os.path.join(output_directory, 'test_tensordataset.pt')
    torch.save(test_data, test_tensors_path)
