'''
Module
'''

###------IMPORTING LIBRARIES-------###
import pandas as pd
import numpy as np
import os
import re
from glob import glob
import mne
import matplotlib.pyplot as plt
import ruptures as rpt
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array
import gc
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

###------FORMATTING TS2VEC INPUT------###
def formatting_ts2vec_input(
    save_path, 
    suffix,
    preprocessed_meps_dir,
    outcomes,
    channels,
    time_col='time',
    value_col='value',
    id_col='id',
    participant_col='participant_id',
    timing='direct',
    ):
    '''
    Function to transform preprocessed MEPs to input required for TS2Vec.

    Input:
        save_path (path)                                    : Path to the folder the data is saved to
        suffix (str)                                        : Define extremity the data belongs to
        preprocessed_meps_dir (path)                        : Path to folder in which the preprocessed MEPs have been saved
        outomes (pd.DataFrame)                              : Dataframe containing outcome labels
        channels (list)                                     : List containing channels from the extremity
        time_col, value_col, id_col, participant_col (str)  : Specified columns
        timing (str)                                        : Timing of outcome you're interested in
    Output:
        X                                                   : Data for input TS2vec (n_epochs, n_timestamps, n_channels)
        epoch_ids                                           : Array containing the epoch_ids
        epoch_patient_labels                                : Array containing the patient labels
        epoch_outcome_labels                                : Array containing the outcome labels
    '''

    # Create paths
    path_to_data    = os.path.join(save_path, f'X_{suffix}.npy')
    path_to_labels  = os.path.join(save_path, f'epoch_patient_labels_{suffix}.npy')
    path_to_epochs  = os.path.join(save_path, f'epoch_ids_{suffix}.npy')

    if timing == 'direct':
        path_to_outcome_labels = os.path.join(save_path, f'epoch_outcome_labels_{suffix}.npy')
    elif timing == '3m':
        path_to_outcome_labels = os.path.join(save_path, f'epoch_outcome_labels_3m_{suffix}.npy')
    else:
        path_to_outcome_labels = os.path.join(save_path, f'epoch_outcome_labels_3m_{suffix}.npy')
        
    # Remove patients
    remove_ids = ['EMC-029','EMC-033','EMC-034','EMC-036','EMC-037','EMC-040',
        'EMC-041','EMC-042','EMC-043','EMC-045','EMC-047','EMC-048','EMC-049',
        'SMART-019','SMART-119','SMART-159','SMART-160','SMART-161','SMART-161','SMART-163',
        'SMART-164','SMART-165','SMART-167','SMART-168','SMART-170','SMART-171',
        'SMART-172', 'SMART-150', 'SMART-151','SMART-152','SMART-153','SMART-154',
        'SMART-155','SMART-156','SMART-157']
    
    # Create dictionary containing dataframes with preprocessed MEPs
    muscle_dfs = {}
    for ch in channels:
        df_ch           = pd.read_csv(os.path.join(preprocessed_meps_dir, f'preprocessed_MEP_{ch}.csv'))
        df_ch           = df_ch[~df_ch[participant_col].isin(remove_ids)] 
        muscle_dfs[ch]  = df_ch 
        del df_ch

    epoch_ids               = sorted(muscle_dfs[channels[0]][id_col].unique())
    max_len                 = 0 #initialize max length of epoch
    rel_times_by_epoch_ch   = {}
    valid_epoch_ids         = []

    # Create relative timing
    for epoch_id in epoch_ids:
        skip_epoch = False

        for ch in channels:
            df_epoch = (
                muscle_dfs[ch].loc[muscle_dfs[ch][id_col] == epoch_id].sort_values(time_col)
            )
            if len(df_epoch) == 0:
                print(f'LEEG: epoch_id: {epoch_id}, channel={ch}')
                skip_epoch=True
                continue
        
        if skip_epoch:
            continue
            
        valid_epoch_ids.append(epoch_id)
        for ch in channels:
            df_epoch    = (
                muscle_dfs[ch].loc[muscle_dfs[ch][id_col] == epoch_id].sort_values(time_col)
            )
            times       = df_epoch[time_col].to_numpy()
            rel_times   = times-times[0] # Create relative time -> each epoch starts at t=0

            rel_times_by_epoch_ch[(epoch_id, ch)] = rel_times # Save relative times
            max_len     = max(max_len, len(df_epoch)) # Save max_len -> is used to padd all epochs to the same length
    
    # Initialize empty array for TS2Vec input (n_epochs, n_timestamps, n_channels)
    epoch_ids   = valid_epoch_ids # Save only epochs containing all channels
    X           = np.full((len(epoch_ids), max_len, len(channels)), np.nan, dtype=np.float32)

    time_axes   = {}

    # Fill in TS2Vec array
    for i, epoch_id in enumerate(epoch_ids):
        time_axes[epoch_id] = {}

        for j, ch in enumerate(channels):
            df_epoch = (
                muscle_dfs[ch].loc[muscle_dfs[ch][id_col]==epoch_id].sort_values(time_col)
            )
            if len(df_epoch) == 0:
                continue

            values      = df_epoch[value_col].to_numpy(dtype=np.float32)
            rel_times   = rel_times_by_epoch_ch[(epoch_id,ch)]

            X[i, :len(values), j]   = values
            time_axes[epoch_id][ch] = rel_times
    
    # Create array from patient labels to use for coloring the embeddings
    print('labels maken voor epochs: patieent id + outcome labels')
    epoch_to_patient = {
        epoch_id: muscle_dfs[channels[0]].loc[
            muscle_dfs[channels[0]][id_col] == epoch_id, participant_col
        ].iloc[0]
        for epoch_id in epoch_ids
        if len(muscle_dfs[channels[0]].loc[muscle_dfs[channels[0]][id_col] == epoch_id])
    }
    epoch_patient_labels = np.array([epoch_to_patient[epoch_id] for epoch_id in epoch_ids if epoch_id in epoch_to_patient])

    # Create array from outcome labels to use for coloring the embeddings
    outcome_map          = outcomes.set_index('participant_id')[f'outcome_label_{suffix}'].to_dict()
    epoch_outcome_labels = np.array([outcome_map[pid] for pid in epoch_patient_labels])

    # Save data
    np.save(path_to_data, X)
    np.save(path_to_labels, epoch_patient_labels)
    np.save(path_to_outcome_labels, epoch_outcome_labels)
    np.save(path_to_epochs, np.array(epoch_ids))

    return X, epoch_ids, epoch_patient_labels, epoch_outcome_labels