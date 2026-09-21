'''
This module contains functions to load MEP data from correct folders, 
and to preprocess the MEP data. This is the preprocessing pipeline for
both the supervised machine learning model and the contrastive learning model.

Author: B.C. Burema
Date: 07/08/2026
'''

###------IMPORTING LIBRARIES------###
import pandas as pd
import numpy as np
import os
import re
from glob import glob
import mne
import matplotlib.pyplot as plt
import ruptures as rpt
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler
from sklearn.utils.validation import check_array
from TM3_Internship_BB.preprocessing.visualization import plot_number_epochs
import gc


###------LOADING DATA------###
def mrc_to_int(series):
    '''
    Change MRC from string to integer.
    Input 
        series (series) :     Series of strings of MRC value.
    Ouput:
        (series)        :     Series of integers of MRC value.
    '''
    return (series.astype(str).str.extract(r'(\d+)')[0].astype('Int32'))

def load_excel(csv_paths, key_path, threshold, timing):
    '''
    Loading clinical characteristics and postoperative outcomes of participants.
    Calculate outcome labels per body part: delta mrc = postoperative mrc - preoperative mrc.
        delta mrc < 0 indicates deterioration, so outcome label = 1.

    Input:
        csv_paths (list)            :   List containing paths to CSV files containing clinical characteristics
                                        and postoperative outcomes of participants of both EMC and GOSH.
        key_path (str)              :   Path to csv file containing keys for GOSH data.
        threshold (int)             :   Threshold to determine outcome label.
        timing (str)                :   Timing of postoperative assessment you're interested in. 
                                        Choose from: 'direct', '3m', '12m'.
    Output:
        clinical_info (pd.DataFrame):   Dataframe containing clinical characteristics 
                                        and postoperative outcomes of participants.
        outcomes_mrc (pd.DataFrame) :   Dataframe containing outcome labels for training 
                                        -> timing of outcomes depends on input.
    '''

    clinical_dfs = []

    for csv_path in csv_paths:
        # Read CSV file and rename column containing Participant ID
        df                   = pd.read_csv(csv_path, sep=';')
        df                   = pd.DataFrame(df).rename(columns={'Participant Id':'participant_id'})
        clinical_dfs.append(df)

    # Combine clinical information of GOSH and EMC
    clinical_info = pd.concat(clinical_dfs, axis=0, ignore_index=True)

    # Change GOSH participant ids in a way these align with participant ids in data folder
    key = pd.read_excel(key_path)
    key['CASTOR number'] = (key['CASTOR number'].astype(str).str.strip())   # Clinical participant id
    key['SMART nmbr'] = (key['SMART nmbr'].astype(str).str.strip())         # Data participant id
    
    gosh_to_smart = dict(zip(
        key['CASTOR number'],
        key['SMART nmbr']
    ))

    clinical_info['participant_id'] = clinical_info['participant_id'].apply(
        lambda x: gosh_to_smart[x] if x.startswith('GOSH-') and x in gosh_to_smart else x
    )
    
    # Depending on timing -> determine outcome_labels -> create dataframe outcomes_mrc
    if timing == 'direct':
        save_loc = r'Z:\14_MMC_IONM_ML\3-Experiments\1-Inputs\outcomes_mrc.csv' 
        if os.path.exists(save_loc):
            print(f'A dataframe with mrc outcomes already exists')
            outcomes_mrc = pd.read_csv(save_loc, sep=',')
        else:
            outcomes_mrc = pd.DataFrame() 

            outcomes_mrc['participant_id']                  = clinical_info['participant_id']
            outcomes_mrc['pre_operative_mrc_right_leg']     = mrc_to_int(clinical_info['pre_operative_mrc_score_of_the_right_leg'])
            outcomes_mrc['pre_operative_mrc_right_foot']    = mrc_to_int(clinical_info['pre_operative_mrc_score_of_the_right_foot'])
            outcomes_mrc['pre_operative_mrc_left_leg']      = mrc_to_int(clinical_info['pre_operative_mrc_score_of_the_left_leg']) 
            outcomes_mrc['pre_operative_mrc_left_foot']     = mrc_to_int(clinical_info['pre_operative_mrc_score_of_the_left_foot']) 

            outcomes_mrc['post_operative_mrc_right_leg']    = mrc_to_int(clinical_info['post_operative_mrc_score_of_the_right_leg_immediately_after_surg'])
            outcomes_mrc['post_operative_mrc_right_foot']   = mrc_to_int(clinical_info['post_operative_mrc_score_of_the_right_foot_immediately_after_sur'])
            outcomes_mrc['post_operative_mrc_left_leg']     = mrc_to_int(clinical_info['post_operative_mrc_score_of_the_left_leg_immediately_after_surge'])
            outcomes_mrc['post_operative_mrc_left_foot']    = mrc_to_int(clinical_info['post_operative_mrc_score_of_the_left_foot_immediately_after_surg'])

            outcomes_mrc['delta_mrc_right_leg']             = outcomes_mrc['post_operative_mrc_right_leg'] - outcomes_mrc['pre_operative_mrc_right_leg']
            outcomes_mrc['delta_mrc_right_foot']            = outcomes_mrc['post_operative_mrc_right_foot'] - outcomes_mrc['pre_operative_mrc_right_foot']
            outcomes_mrc['delta_mrc_left_foot']             = outcomes_mrc['post_operative_mrc_left_foot'] - outcomes_mrc['pre_operative_mrc_left_foot']
            outcomes_mrc['delta_mrc_left_leg']              = outcomes_mrc['post_operative_mrc_left_leg'] - outcomes_mrc['pre_operative_mrc_left_leg']

            # Create outcome label
            # delta_mrc < 0 indicates deterioration, so outcome label = 1
            outcomes_mrc['outcome_label_right_leg']         = (outcomes_mrc['delta_mrc_right_leg'] <= threshold).astype(int)  
            outcomes_mrc['outcome_label_right_foot']        = (outcomes_mrc['delta_mrc_right_foot'] <= threshold).astype(int) 
            outcomes_mrc['outcome_label_left_leg']          = (outcomes_mrc['delta_mrc_left_leg'] <= threshold).astype(int) 
            outcomes_mrc['outcome_label_left_foot']         = (outcomes_mrc['delta_mrc_left_foot'] <= threshold).astype(int) 
            
            outcomes_mrc.to_csv(save_loc, index=False)
    elif timing == '3m':
        save_loc = r'Z:\14_MMC_IONM_ML\3-Experiments\1-Inputs\outcomes_mrc_3m.csv' 
        if os.path.exists(save_loc):
            print(f'A dataframe with mrc outcomes already exists')
            outcomes_mrc = pd.read_csv(save_loc, sep=',')
        else:
            outcomes_mrc = pd.DataFrame() 

            outcomes_mrc['participant_id']                  = clinical_info['participant_id']
            outcomes_mrc['pre_operative_mrc_right_leg']     = mrc_to_int(clinical_info['pre_operative_mrc_score_of_the_right_leg'])
            outcomes_mrc['pre_operative_mrc_right_foot']    = mrc_to_int(clinical_info['pre_operative_mrc_score_of_the_right_foot'])
            outcomes_mrc['pre_operative_mrc_left_leg']      = mrc_to_int(clinical_info['pre_operative_mrc_score_of_the_left_leg']) 
            outcomes_mrc['pre_operative_mrc_left_foot']     = mrc_to_int(clinical_info['pre_operative_mrc_score_of_the_left_foot']) 

            outcomes_mrc['post_operative_mrc_right_leg']    = mrc_to_int(clinical_info['post_operative_mrc_score_of_the_right_leg_at_3_months'])
            outcomes_mrc['post_operative_mrc_right_foot']   = mrc_to_int(clinical_info['post_operative_mrc_score_of_the_right_foot_at_3_months'])
            outcomes_mrc['post_operative_mrc_left_leg']     = mrc_to_int(clinical_info['post_operative_mrc_score_of_the_left_leg_at_3_months'])
            outcomes_mrc['post_operative_mrc_left_foot']    = mrc_to_int(clinical_info['post_operative_mrc_score_of_the_left_foot_at_3_months'])

            outcomes_mrc['delta_mrc_right_leg']             = outcomes_mrc['post_operative_mrc_right_leg'] - outcomes_mrc['pre_operative_mrc_right_leg']
            outcomes_mrc['delta_mrc_right_foot']            = outcomes_mrc['post_operative_mrc_right_foot'] - outcomes_mrc['pre_operative_mrc_right_foot']
            outcomes_mrc['delta_mrc_left_foot']             = outcomes_mrc['post_operative_mrc_left_foot'] - outcomes_mrc['pre_operative_mrc_left_foot']
            outcomes_mrc['delta_mrc_left_leg']              = outcomes_mrc['post_operative_mrc_left_leg'] - outcomes_mrc['pre_operative_mrc_left_leg']

            # Create outcome label
            # delta_mrc < 0 indicates deterioration, so outcome label = 1
            outcomes_mrc['outcome_label_right_leg']         = (outcomes_mrc['delta_mrc_right_leg'] <= threshold).astype(int)  
            outcomes_mrc['outcome_label_right_foot']        = (outcomes_mrc['delta_mrc_right_foot'] <= threshold).astype(int) 
            outcomes_mrc['outcome_label_left_leg']          = (outcomes_mrc['delta_mrc_left_leg'] <= threshold).astype(int) 
            outcomes_mrc['outcome_label_left_foot']         = (outcomes_mrc['delta_mrc_left_foot'] <= threshold).astype(int) 
            
            outcomes_mrc.to_csv(save_loc, index=False)
    else:
        save_loc = r'Z:\14_MMC_IONM_ML\3-Experiments\1-Inputs\outcomes_mrc_12m.csv' 
        if os.path.exists(save_loc):
            print(f'A dataframe with mrc outcomes already exists')
            outcomes_mrc = pd.read_csv(save_loc, sep=',')
        else:
            outcomes_mrc = pd.DataFrame() 

            outcomes_mrc['participant_id']                  = clinical_info['participant_id']
            outcomes_mrc['pre_operative_mrc_right_leg']     = mrc_to_int(clinical_info['pre_operative_mrc_score_of_the_right_leg'])
            outcomes_mrc['pre_operative_mrc_right_foot']    = mrc_to_int(clinical_info['pre_operative_mrc_score_of_the_right_foot'])
            outcomes_mrc['pre_operative_mrc_left_leg']      = mrc_to_int(clinical_info['pre_operative_mrc_score_of_the_left_leg']) 
            outcomes_mrc['pre_operative_mrc_left_foot']     = mrc_to_int(clinical_info['pre_operative_mrc_score_of_the_left_foot']) 

            outcomes_mrc['post_operative_mrc_right_leg']    = mrc_to_int(clinical_info['post_operative_mrc_score_of_the_right_leg_at_12_months'])
            outcomes_mrc['post_operative_mrc_right_foot']   = mrc_to_int(clinical_info['post_operative_mrc_score_of_the_right_foot_at_12_months'])
            outcomes_mrc['post_operative_mrc_left_leg']     = mrc_to_int(clinical_info['post_operative_mrc_score_of_the_left_leg_at_12_months'])
            outcomes_mrc['post_operative_mrc_left_foot']    = mrc_to_int(clinical_info['post_operative_mrc_score_of_the_left_foot_at_12_months'])

            outcomes_mrc['delta_mrc_right_leg']             = outcomes_mrc['post_operative_mrc_right_leg'] - outcomes_mrc['pre_operative_mrc_right_leg']
            outcomes_mrc['delta_mrc_right_foot']            = outcomes_mrc['post_operative_mrc_right_foot'] - outcomes_mrc['pre_operative_mrc_right_foot']
            outcomes_mrc['delta_mrc_left_foot']             = outcomes_mrc['post_operative_mrc_left_foot'] - outcomes_mrc['pre_operative_mrc_left_foot']
            outcomes_mrc['delta_mrc_left_leg']              = outcomes_mrc['post_operative_mrc_left_leg'] - outcomes_mrc['pre_operative_mrc_left_leg']

            # Create outcome label
            # delta_mrc < 0 indicates deterioration, so outcome label = 1
            outcomes_mrc['outcome_label_right_leg']         = (outcomes_mrc['delta_mrc_right_leg'] <= threshold).astype(int)  
            outcomes_mrc['outcome_label_right_foot']        = (outcomes_mrc['delta_mrc_right_foot'] <= threshold).astype(int) 
            outcomes_mrc['outcome_label_left_leg']          = (outcomes_mrc['delta_mrc_left_leg'] <= threshold).astype(int) 
            outcomes_mrc['outcome_label_left_foot']         = (outcomes_mrc['delta_mrc_left_foot'] <= threshold).astype(int) 
            
            outcomes_mrc.to_csv(save_loc, index=False)
    return clinical_info, outcomes_mrc

def load_times_epochs(metadata_path, participant_ids, records, institution):
    '''
    Load information about start times of MEP recordings, after baseline establishment.

    Input:
        metadata_path (str)                 :   Path to folder containing folders with metadata per participant.
        participant_ids (set)               :   Set containing participant_ids of patients with both MEP recordings and clinical information.
        records (list)                      :   List containing dictionaries per participant.
                                                Each dictionary contains participant_id, epochs (EpochsFIF), number of epochs. 
        institution (str)                   :   Either 'EMC' or 'GOSH' -> necessary to define, because of method to determine the first epoch.
    Output:
        start_times (pd.DataFrame)          :   Dataframe containing columns participant_id, start_idx, and start_time.
                                                Start_idx is used to remove first couple of epochs used for baseline establishment.
        surgical_durations (pd.DataFrame)   :   Dataframe containing columns participant_id, surgical duration.
    '''

    start_times = []
    surgical_durations = []

    if institution == 'EMC':
        for participant_folder in os.listdir(metadata_path):
            # Create path per participant within the metadata folder
            participant_path = os.path.join(metadata_path, participant_folder)

            participant_id = participant_folder.split('_')[-1]

            if participant_id in participant_ids:
                # Search for TRG files containing metadata information
                trg_files = glob(os.path.join(participant_path, "*.TRG.csv"))
            
                if len(trg_files) == 0:
                    print(f'No TRG file found for participant {participant_id}')
                    continue
                
                meta_df = pd.read_csv(trg_files[0], sep=',', header=4)
                
                meta_df.columns = meta_df.columns.str.strip()
                # Use one unique muscle to retrieve information about timing of recordings
                meta_df = meta_df[meta_df["Tr No"] == 7] 
                
                # Change data format of Date Time to datetime
                meta_df['Date Time'] = meta_df['Date Time'].str.strip().str.replace('"','').str.replace(' "','').str.replace('" ','')
                meta_df['Date Time'] = pd.to_datetime(
                                            meta_df['Date Time'],
                                            dayfirst=True)

                unique_timestamps = meta_df['Date Time'].drop_duplicates().reset_index(drop=True)

                # Search for the time gap of 20 minutes between successive epochs
                diff_time = unique_timestamps.diff()
                threshold = 20
                gap_idx = diff_time[diff_time > pd.Timedelta(minutes=threshold)].index
                start_session_idx = gap_idx[0]
                start_time = unique_timestamps.iloc[start_session_idx]

                start_times.append({
                    'participant_id': f'EMC-{participant_id}',
                    'start_idx': start_session_idx,
                    'start_time': start_time
                })

                end_time = unique_timestamps.iloc[-1]
                surgical_duration = end_time-start_time

                surgical_durations.append({
                    'participant_id': f'EMC-{participant_id}',
                    'surgical_duration': surgical_duration
                })

        surgical_durations = pd.DataFrame(surgical_durations)
        start_times = pd.DataFrame(start_times)
    
    if institution == 'GOSH':
        for record in records:
            participant_id = record['participant_id']
            metadata = record['epochs'].metadata.reset_index(drop=True)

            # Manually get start_session_idx for participant SMART-150. This is the only patient for whom this method did not apply.
            if participant_id == 'SMART-150':
                start_session_idx = 72
                start_time = metadata.loc[start_session_idx,'epoch_timestamp']
                end_time = pd.to_datetime(metadata.loc[len(metadata)-1, 'epoch_timestamp'])
               
                start_times.append({
                    'participant_id': f'{participant_id}',
                    'start_idx': start_session_idx,
                    'start_time': start_time
                })
                surgical_duration = end_time - pd.to_datetime(start_time)

                surgical_durations.append({
                    'participant_id': f'{participant_id}',
                    'surgical_duration': surgical_duration
                })
                continue
            
            # Find indices where annotation_titles is not empty
            annotation_indices = metadata.index[metadata['annotation_titles'].notna() & (metadata['annotation_titles'] != '')].tolist()

            start_session_idx = None

            # Search for the time gap of 15 minutes between successive epochs at timing of annotations. Search for the first annotation at which the time gap is at least 15 minutes.
            for idx in annotation_indices:
                if idx == 0: # if there is no previous epoch to compare to, continue. This cannot be the start of the session.
                    continue
                prev_idx = idx-1
                time_diff = pd.to_datetime(metadata.loc[idx,'epoch_timestamp']) - pd.to_datetime(metadata.loc[prev_idx,'epoch_timestamp'])

                if time_diff.total_seconds() / 60 > 15: 
                    start_session_idx = idx
                    break
            if start_session_idx is None:
                print(f'Warning: start_idx not found for {participant_id}')
                continue
            
            start_time = metadata.loc[start_session_idx,'epoch_timestamp']
            end_time = pd.to_datetime(metadata.loc[len(metadata)-1, 'epoch_timestamp'])
            print(f'start: {start_time}, end: {end_time}')
            start_times.append({
                'participant_id': f'{participant_id}',
                'start_idx': start_session_idx,
                'start_time': start_time
            })
            surgical_duration = end_time - pd.to_datetime(start_time)
            print(f'surgical dur {surgical_duration}')
            surgical_durations.append({
                'participant_id': f'{participant_id}',
                'surgical_duration': surgical_duration
            })
                
        start_times = pd.DataFrame(start_times)
        surgical_durations = pd.DataFrame(surgical_durations)
    return start_times, surgical_durations

def load_data(data_path, metadata_path, clinical_info, file_pattern, id_pattern, institution):
    '''
    Load epochs from participants present in clinical_info dataframe.
    This function can also be applied to different IONM modalities,
    including MEP, SSEP, and BCR.

     Input:
        data_path (str)                     :   Path to folder containing .fif files per participant. 
        metadata_path (str)                 :   Path to folder containing folders with metadata per participant.
        clinical_info (pd.DataFrame)        :   Dataframe containing clinical characteristics 
                                                and postoperative outcomes of participants.
        file_pattern (str)                  :   Pattern used to identify modality-specific files.
        id_pattern (str)                    :   General pattern used to identify participant data.
        institution (str)                   :   Either 'EMC' or 'GOSH' -> necessary to determine, because method to load data differs.
    Output:
        records (list)                      :   List containing dictionaries per participant.
                                                Each dictionary contains participant_id, epochs (EpochsFIF), number of epochs.
        valid_ids (set)                     :   Set containing participant ids of patients with both MEP recordings and clinical information.
        start_epochs (pd.DataFrame)         :   Dataframe containing columns participant_id, start_idx and start_time.
                                                Start_idx is used to remove first couple of epochs used for baseline establishment.
        surgical_durations (pd.DataFrame)   :   Dataframe containing columns participant_id, surgical duration.
    '''
    
    # # Retrieve participant ids of included participants (Sabri)
    if institution == 'EMC':
        valid_ids   = set(clinical_info['participant_id'].astype(str).str.replace('EMC-', ''))
        valid_ids.discard('029') # Remove participant 029 since an error happened during conversion -> data is missing
    elif institution == 'GOSH':
        valid_ids = set(clinical_info['participant_id'].astype(str))
        # valid_ids = ['SMART-138','SMART-158']
    # Initialize list to save records per participant
    records     = [] 

    if institution == 'EMC':
        # Search for epoch files in folder
        files       = glob(os.path.join(data_path, file_pattern))

        # Loop through epoch files to identify those corresponding to valid_ids
        for file in files:
            basename    = os.path.basename(file)
            match       = re.search(id_pattern, basename) 
            participant_id = f'EMC-{match.group(1)}'

            if match and match.group(1) in valid_ids:
                epochs  = mne.read_epochs(file, preload=True,verbose='CRITICAL')

                # Resample the epochs to 3 kHz, so it aligns with GOSH data
                epochs.resample(sfreq=3000) 
                records.append({'participant_id': participant_id,
                                'epochs': epochs,
                                'n_epochs': len(epochs.events)})
    elif institution == 'GOSH':
        # Loop through folders of participants
        for participant_id in os.listdir(data_path):
            participant_path = os.path.join(data_path, participant_id)
            if not os.path.isdir(participant_path):
                continue
            if participant_id not in valid_ids:
                continue
            
            # Search for MEP file within participant folder
            file = os.path.join(participant_path, 'MEP-epo.fif')
            if not os.path.exists(file):
                print(f'Warning: no MEP-epo.fif file found for {participant_id}')
                continue
           
            epochs  = mne.read_epochs(file, preload=True,verbose='CRITICAL')
           
            # Since the epochs from different channels of GOSH patients differ in length, we have to cut all epochs to the shortest duration. 
            data = epochs.get_data()
            valid_lengths = []
            for epoch_idx in range(data.shape[0]):
                epoch_data = data[epoch_idx]                        # (n_channels, n_times)
               
                # Check whether there are epochs containing only NaNs
                fully_nan_ch = np.isnan(epoch_data).all(axis=1)     # (n_channels, ...)
                
                # Check whether there are NaNs at timepoints -> determine length of signal with valid values
                partial_data = epoch_data[~fully_nan_ch] 

                if partial_data.shape[0] == 0:
                    # All channels are NaN in this epoch (edge case) 
                    valid_lengths.append(0)
                    continue

                any_nan_per_timepoint = np.isnan(partial_data).any(axis=0) # (n_times, ..)
                if any_nan_per_timepoint.any():
                    first_nan_idx = np.argmax(any_nan_per_timepoint)
                    valid_lengths.append(first_nan_idx)
                else:
                    valid_lengths.append(data.shape[2])
                
            # Determine minimal length of epochs for this participant
            min_valid_length = min(valid_lengths)
            # Cut all epochs from all channels to this length
            epochs = epochs.copy().crop(tmin=epochs.tmin, tmax=epochs.tmin+(min_valid_length-1) / epochs.info['sfreq'])
            
             # Resample the epochs to 3 kHz, so it aligns with GOSH participant with lowest sample frequency
            epochs.resample(sfreq=3000) 
            records.append({'participant_id': participant_id,
                            'epochs': epochs,
                            'n_epochs': len(epochs.events)})

    start_epochs, surgical_durations = load_times_epochs(metadata_path, valid_ids, records, institution) 
    return records, valid_ids, start_epochs, surgical_durations


###------PREPROCESSING DATA------###
###------Step 1: Removing first epochs
def remove_first_epochs(records, start_epochs):
    '''
    Remove first X epochs based on timing. 
    At least 20 minutes elapse between surgical opening and first MEP recording.

    Input:
        records (list):                 List containing dictionaries per participant.
                                        Each dictionary contains participant_id, epochs (EpochsFIF), number of epochs.
        start_epochs (pd.DataFrame):    Dataframe containing columns participant_id, start_idx, and start_time.
    Output:
        records (list):                 List containing dictionaries per participant.
                                        Each dictionary contains participant_id, epochs (EpochsFIF), number of epochs.
                                        Difference from input: first epochs used for baseline establishment are removed.
    '''
    print('Start removing first epochs')
    for record in records:
        participant_id = record['participant_id']

        epochs = record['epochs'].copy()

        start_idx = start_epochs.loc[start_epochs['participant_id'] == participant_id,'start_idx'].values[0]
 
        epochs = epochs[start_idx:]

        record['epochs']= epochs
    return records

###------Step 2: Removing flat epochs
def flat_epochs_to_nan(records):
    '''
    Change epochs with zero variance to epochs containing NaNs.

    Input:   
        records (list):     List containing dictionaries per participant.
                            Each dictionary contains participant_id, epochs (EpochsFIF), number of epochs.
    Output:
        records (list):     List containing dictionaries per participant.
                            Each dictionary contains participant_id, epochs (EpochsFIF), number of epochs.
                            Difference from input: flat epochs now contain only NaNs.
    '''
    print('Change flat epochs to NaN')
    for record in records:
        # Change MNE.epochsArray to (n_epochs, n_ch, n_time)
        data                = record['epochs'].get_data() 
 
        # Determine variance over time per channel and epoch (n_epochs, n_ch)
        var                 = np.var(data, axis=2) 
        
        # Determine which epochs have zero variance
        flat                = var == 0
        epoch_idx, ch_idx   = np.where(flat)

        # Change epochs with zero variance to epochs containing only NaNs
        for epoch, ch in zip(epoch_idx, ch_idx):
            data[epoch, ch, :] = np.nan
        
        record['epochs']._data = data

    return records

###------Step 3: Renaming channels
def rename_channels(records):
    '''
    Rename channel names to match those of participant 001, as these are the most accurate.

    Input:   
        records (list       :   List containing dictionaries per participant.
                                Each dictionary contains participant_id, epochs (EpochsFIF), number of epochs.
    Output:
        records (list)      :   List containing dictionaries per participant.
                                Each dictionary contains participant_id, epochs (EpochsFIF), number of epochs.
                                Difference from input: Channel names now correspond between participants.
        rec_ch_names (list) :   List containing channel names of participant 001.
    '''
    print('Start renaming channels')
    # Retrieve channel names of participant 001 of EMC. Participants have completely different channel names, but the order of the channels is always the same.
    ref_ch_names     = records[0]['epochs'].info['ch_names'] 
    # Create channel names mapping for GOSH. Most participants have comparable channel names.
    rename_mapping_gosh = {
        'LQD': 'L: QUAD L',
        'QDL': 'L: QUAD L',
        'RQD': 'R: QUAD R',
        'QDR': 'R: QUAD R',
        'LAH': 'L: AH L',
        'AHL': 'L: AH L',
        'RAH': 'R: AH R',
        'AHR': 'R: AH R',
        'LTA': 'L: TA L',
        'TAL': 'L: TA L',
        'RTA': 'R: TA R',
        'TAR': 'R: TA R',
        'LGT': 'L: GAS L',
        'GTL': 'L: GAS L',
        'RGT': 'R: GAS R',
        'GTR': 'R: GAS R'
    }

    # Create list of channel names you're not interested in
    extra_ch_emc = ['R: Sfincter R','L: Sfincter L','L: APB L','R: APB R'] 
    extra_ch_gosh = ['LSP','RSP','LSR','RSR','RAD','LAD','LHM','RHM','LBR','RBR','LAPB','APBL','RAPB','APBR','RBIC','LBIC'] 

    # Rename channel names -> all channel names of all participants are now the same
    for record in records:
        epochs = record['epochs'].copy()

        if 'EMC' in record['participant_id']:
            old_ch_names = epochs.info['ch_names']
            mapping      = dict(zip(old_ch_names, ref_ch_names))
            epochs.rename_channels(mapping)
            epochs.drop_channels(extra_ch_emc)
        else:
            channels_to_drop = [ch for ch in extra_ch_gosh if ch in epochs.ch_names]
            epochs.drop_channels(channels_to_drop)
            mapping = {k: v for k, v in rename_mapping_gosh.items() if k in epochs.ch_names}
            epochs.rename_channels(mapping)
        record['epochs'] = epochs
    
    final_ch_names = records[0]['epochs'].info['ch_names']

    return records, final_ch_names

###------Step 4: Changing data format
def epochs_to_df(records, ch_names):
    '''
    Create a dictionary to store data per channel (muscle). 
    This dictionary contains epochs in DataFrame format, with columns 'id', 'sfreq', 'time', and 'values'.
        - The 'id' is created by combining participant ID and number of epoch. Example: '001_1'.

    Input:
        records (list)      :   List containing dictionaries per participant.
                                Each dictionary contains participant_id, epochs (EpochsFIF), number of epochs.
        ch_names (list)     :   List containing channel names of participant 001.    
    Output:
        muscle_dfs (dict)   :   Dictionary containing dataframes per muscle. 
    '''
    print('Start formatting epochs to df')
    # Initialize dictionary to store dataframes of MEP epochs.
    muscle_dfs = {ch: [] for ch in ch_names}

    # Loop trough records of individual participants to convert data to dataframes
    for record in records:
        participant_id = record['participant_id']
        print(f'Processing participant {participant_id}')
        sample_f       = record['epochs'].info['sfreq']

        # Loop through channels to save data per channel into dataframe
        for ch_name in ch_names:
            if ch_name not in record['epochs'].ch_names: 
                print(f'for participant {participant_id} this channel is not present: {ch_name}')
                continue 
            
            muscle_df = record['epochs'].to_data_frame(picks=[ch_name])

            # Reduce workload
            float_cols = muscle_df.select_dtypes(include='float64').columns
            muscle_df[float_cols] = muscle_df[float_cols].astype('float32')
            int_cols = muscle_df.select_dtypes(include='int64').columns
            muscle_df[int_cols] = muscle_df[int_cols].astype('int32')

            # Remove first of double trains for specific participants -> visually assessed.
            if participant_id           == 'EMC-005':
                tmin                    = 0.002
                muscle_df               = muscle_df[(muscle_df['time']  >=tmin)]

                # Change first timestemp to zero
                muscle_df['time']       = muscle_df['time']-muscle_df['time'].min()            
            if participant_id           == 'EMC-014': 
                # For participant 014, there is a difference in stimulus artifact on the left and right side
                if 'R' in ch_name:
                    tmin                = 0.018
                    muscle_df           = muscle_df[(muscle_df['time']  >=tmin)]
                    
                    # Change first timestemp to zero
                    muscle_df['time']   = muscle_df['time']-muscle_df['time'].min() 
                else:
                    tmin                = 0.010
                    muscle_df           = muscle_df[(muscle_df['time']  >=tmin)]

                    # Change first timestemp to zero
                    muscle_df['time']   = muscle_df['time']-muscle_df['time'].min()

            if participant_id           == 'EMC-015':
                tmin                    = 0.016
                muscle_df               = muscle_df[(muscle_df['time']  >=tmin)]

                # Change first timestemp to zero
                muscle_df['time']       = muscle_df['time']-muscle_df['time'].min()

            if participant_id           == 'EMC-016':
                tmin                    = 0.008
                muscle_df               = muscle_df[(muscle_df['time']  >= tmin)]

                # Change first timestemp to zero
                muscle_df['time']       = muscle_df['time']-muscle_df['time'].min()
            if participant_id           == 'SMART-149':
                tmin                    = 0.02
                muscle_df               = muscle_df[(muscle_df['time'] > tmin)]
                muscle_df['time']       = muscle_df['time']-muscle_df['time'].min()
            if 'SMART' in participant_id:
                tmin                    = 0.015
                muscle_df               = muscle_df[(muscle_df['time'] > tmin)]
                muscle_df['time']       = muscle_df['time']-muscle_df['time'].min()

            # Create an id per epoch: combination of participant_id and number of epoch. Example: 'EMC-001_1'.
            muscle_df['id']             = participant_id + '_' + muscle_df['epoch'].astype(str)
            muscle_df['sfreq']          = sample_f

            # # Create a key for a channel name if it does not yet exist in the dictionary
            muscle_dfs[ch_name].append(muscle_df)
            del muscle_df
        gc.collect()
   
    # Concatenate all epoch dataframes per muscle into a single dataframe
    # -> Key = channel_name, value = dataframe containing all epochs from all participants for that channel
    for ch_name in muscle_dfs:
        muscle_dfs[ch_name] = pd.concat(muscle_dfs[ch_name], ignore_index=True)
    return muscle_dfs

def get_channel_name(key):
    '''
    Retrieve and clean name of muscles.

    Input:
        key (str)   :   Channel name
    Output:
        key (str)   :   Clean channel name
    '''

    # Remove L: or R: before name of muscle
    key = key.replace('L: ', '').replace('R: ', '').strip()

    parts = key.split()
    if parts[-1] in ['L','R']:
        parts = parts[:-1]
    return ' '.join(parts)

def merge_channels_left_right(muscle_dict, data_path):
    '''
    Merge left/right channels of sfincter into one DataFrame. 
        Ensure that all epochs include an indication of left or right or right side.
        Save dataframes as csv.

    Input:
        muscle_dict (dict)  :   Dictionary containing dataframes per muscle.
        data_path (str)     :   Path to folder in which the dataframes are saved.
    '''
    
    # Left and right sides are kept separate for leg and foot muscles, but are combined for sfincter. 
    ch_names = set(get_channel_name(k) for k in muscle_dict.keys())

    # Loop through channel names
    for ch_name in ch_names:
        print(f'Start merging {ch_name}')
        # Clean channel names
        l_key = next((k for k in muscle_dict if k.startswith('L:') and get_channel_name(k)==ch_name), None)
        r_key = next((k for k in muscle_dict if k.startswith('R:') and get_channel_name(k)==ch_name), None)

        if l_key:
            # Create key for left side
            df_ch = muscle_dict.pop(l_key).rename(columns={l_key: 'value'}, copy=False)
            save_path = os.path.join(data_path, f'MEP_{ch_name}_L.csv')
            df_ch.to_csv(save_path, index=False)
            del df_ch
        if r_key:
            # Create key for right side
            df_ch = muscle_dict.pop(r_key).rename(columns={r_key: 'value'}, copy=False)
            save_path = os.path.join(data_path, f'MEP_{ch_name}_R.csv')
            df_ch.to_csv(save_path, index=False)
            del df_ch
        gc.collect()
       
###------Step 5: Remove stimulus artifact
def crop_stimart(ch, df, plot=False):
    '''
    Detect the changepoint between stimulus artifact and the remaining signal,
    using changepoing analysis for binary segmenatation. Identify changepoint based 
    on changes in both mean and variance.

    Input:
        ch (str)                :   Channel name
        df (pd.DataFrame)       :   Dataframe containing epochs form all participants for a single channel.
        plot (bool)             :   Whether to plot the changepoints or not.
    Output:
        indices_to_drop (list)  :   List of indices to drop in dataframe to remove stimulus artifact in each epoch
    '''
    print(f'Start changepoint analysis for channel {ch}')
    # Define dict to save changepoints per participant
    participant_changepoints = {}

    # Loop through epochs to determine changepoint per epoch
    for id, group in df.groupby('id'):
        participant_id  = id.split('_')[0]

        # Define timing until which the changepointanalysis is performed -> visually assessed
        if 'EMC' in participant_id:
            search_t = 0.021
        else:
            participant_id_check = int(participant_id.split('-')[1])
            if participant_id == 21:
                search_t = 0.02
            elif participant_id in [103,107,108,116,122,139,143,144,147,149,152,153,164]:
                search_t = 0.015
            elif participant_id_check >= 100: 
                search_t = 0.025 
            else:
                signal_length = group['time'].max()  
                search_t = 0.025 if signal_length > 0.14 else 0.015

        # Get values of data segment up until search_t
        signal          = group['value'].values
        segment         = group[(group['time'] <= search_t)]
        data_segment    = segment['value'].values
        sample_f       = group['sfreq'].values[0]

        # Use Binary Segmentation to identify the changepoint between stimulus artifact and remaining signal.
        # Binary segmentation was chosen because one clear change is present in the signal.
        # Model = 'normal' was chosen because there is mainly a change in mean and variance.
        algo = rpt.Binseg(model='normal').fit(data_segment)
        cps  = algo.predict(n_bkps=1)
        cps  = cps[:-1]

        # Plot epoch and detected changepoint
        if plot:
            if id == 'EMC-001_9':
                time_ms = np.arange(len(signal)) / sample_f * 1000
                plt.figure(figsize=(12,4))
                plt.plot(time_ms, signal, color='black',linewidth=0.8)
                for cp in cps:
                    plt.axvline(cp/sample_f*1000, linestyle='--',color='red', linewidth=1, label='Changepoint')
                plt.xlabel('Time [ms]')
                plt.ylabel('Amplitude [uV]')
                plt.title(f'Original MEP {ch} with detected changepoint')
                plt.legend()
                plt.tight_layout()
                plt.savefig(fr'Z:\14_MMC_IONM_ML\6-Analysis_results\TM3_Internship_BB/changepoint_{id}.png')
                plt.close()

        # Store changepoints of each epoch per participant
        if participant_id not in participant_changepoints:
            participant_changepoints[participant_id] = []
        participant_changepoints[participant_id].append((id, cps[0]))
        
    # Store changepoint at which the epochs of each participant need to be cropped
    # This is defined as the median of all identified changepoints for a given participant within each channel
    changepoints_to_crop = {}
    for participant_id, cp_list in participant_changepoints.items():
        cp_values = [cp for _, cp in cp_list]

        # Determine the median of all identified changepoint of a given participant
        cp_crop = np.nanpercentile(cp_values, 50)
        print(f'Participant {participant_id}, channel {ch} changepoint at {cp_crop/sample_f*1000} ms')
        
        # After visual inspection, the median changepoint of one participant within one channel was found to be 
        # incorrectly identified and was manually corrected.
        if participant_id == 'EMC-005' and ch == 'GAS_L': 
            cp_crop = 140

        # Store median changepoint per participant -> this is the point at which the signal will be cropped.
        changepoints_to_crop[participant_id] = int(round(cp_crop))

    # Create a list with indices to drop per epoch
    indices_to_drop = []
    for id, group in df.groupby('id'):
        participant_id  = id.split('_')[0]
        cp_crop         = changepoints_to_crop[participant_id]
        indices_to_drop.extend(group.iloc[:cp_crop].index) 
    
    return indices_to_drop

###------Step 6: Scaling
def scale_group(group, scaler, plot=False):
    '''
    Fit and transform scaler on data of a given participant for a single channel.

    Input:
        group (pd.DataFrame)                    :   Dataframe containing epochs of a given participant for a single channel
        scaler                                  :   Initialized scaler
        plot (bool)                             :   Whether to plot the scaled epochs or not
    Output:
        group (pd.DataFrame)                    :   Dataframe containing epochs of a given participant for a single channel.
                                                    Difference from input: all epochs are scaled.

    '''

    group = group.copy()

    # Fit and transform data per participant per channel. 
    # It is allowed to fit and transform at once, since we perform scaling per participant per channel. No risk of data leakage.
    group['value_scaled'] = scaler.fit_transform(group[['value']])
   
    if plot: 
        participant_id = group['id'].iloc[0].split('_')[0]  
        epoch_id = 'EMC-001_9'

        if epoch_id in group['id'].values:
            epoch = group[group['id']==epoch_id].sort_values('time')

            time_ms = epoch['time'].to_numpy() * 1000
            scaled = epoch['value_scaled'].to_numpy()

            plt.figure(figsize=(12,4))
            plt.plot(time_ms, scaled, color='black',linewidth=0.8)

            plt.xlabel('Time [ms]')
            plt.ylabel('Amplitude [uV]')
            plt.title(f'Scaled MEP')
            plt.legend()
            plt.tight_layout()
            plt.savefig(r'Z:\14_MMC_IONM_ML\6-Analysis_results\TM3_Internship_BB/scaled.png')
            plt.close()
    return group 

def scale_epochs(df, plot=False):
    '''
    Scale epochs using 5th and 95th percentile of MEP amplitude, determined per participant per channel.

    Input:
        df (pd.DataFrame)        :   Dataframe containing epochs form all participants for a single channel.
        plot (bool)              :   Whether to plot the scaled epochs or not.
    Output:
        df_scaled (pd.DataFrame) :   Dataframe containing scaled epochs from all participants for a single channel.
    '''
    print('start scaling')
    # Retrieve participant id -> necessary to determine 5th and 95th percentile of MEP data of a given participant for a given channel
    df['participant_id'] = df['id'].str.split('_').str[0]

    # Initialize scaler
    scaler = RobustScaler(quantile_range=(5,95))

    # Group data per participant and apply scaler
    df_scaled = df.groupby('participant_id').apply(lambda group: scale_group(group,scaler,plot))
    del df
    df_scaled.reset_index(drop=True, inplace=True)
    df_scaled['value'] = df_scaled['value'].astype('float32')
    df_scaled['value_scaled'] = df_scaled['value_scaled'].astype('float32')
    return df_scaled

###------Combine first 4 steps of preprocessing
def create_df_per_ch(records, start_idx_epochs, data_path):
    '''
    Combine first preprocessing steps into a single function.
    Running this funciton alone is sufficient to perform first part of preprocessing of all data.
    Dataframes per muscle are stored.

    Input:
        records (list)                 :    List containing dictionaries per participant with original data.
                                            Each dictionary contains participant_id, epochs (EpochsFIF), number of epochs.
        start_idx_epochs (pd.Dataframe):    Dataframe containing columns participant_id, start_idx and start_time.
                                            Start_idx is used to remove first couple of epochs used for baseline establishment.
        data_path (str)                :    Path to folder in which the dataframes are saved.
    '''

    # Step 1: Remove first epochs
    preprocessed_records = remove_first_epochs(records, start_idx_epochs)
    # Step 2a: Change flat epochs to NaNs
    preprocessed_records = flat_epochs_to_nan(preprocessed_records)
    
    # Step 3: Rename channel names
    preprocessed_records, ch_names = rename_channels(preprocessed_records)

    # Step 4: Change dataformat from mne.EpochArray to dataframe
    preprocessed_records        = epochs_to_df(preprocessed_records, ch_names)

    # Merge left and right dataframes for sfincter -> epoch_id was changed to 001_1_L or 001_1_R depending on the recording side
    # Keep left and right dataframes separated for all other muscles.
    # Merge per spier -> sla per spier de df op als csv
    merge_channels_left_right(preprocessed_records, data_path)

###------Combine final 3 steps of preprocessing and create dataframe per channel
def preprocessing(ch, df, preprocessed_data_path):
    '''
    Combine final preprocessing steps into a single function.
    Running this funciton alone is sufficient to perform final part of preprocessing of all data.
    Dataframes with preprocessed epcohs per muscle are stored.

    Input:
        records (list)                 :    List containing dictionaries per participant with original data.
                                            Each dictionary contains participant_id, epochs (EpochsFIF), number of epochs.
        start_idx_epochs (pd.Dataframe):    Dataframe containing columns participant_id, start_idx and start_time.
                                            Start_idx is used to remove first couple of epochs used for baseline establishment.
        data_path (str)                :    Path to folder in which the dataframes are saved.
    '''
    # Step 2b: Remove epochs with only NaNs
    df.dropna(subset=['value'],inplace=True)

    # Step 5: Remove stimulus artifacts
    # Determine changepoints per participant per channel, and use these changepoints to crop stimulus artifact   
    indices_to_drop = crop_stimart(ch, df, plot=True)
    df.drop(indices_to_drop, inplace=True)

    # Step 6: Scaling
    # Scale epochs using the 5th and 95th percentiles computed per channel for each participant
    # This scaling changes the scale to [-1, 1]
    df['value'] = df['value'].astype('float32')
    df = scale_epochs(df,plot=True)

    save_path = os.path.join(preprocessed_data_path, f'preprocessed_MEP_{ch}.csv')
    df.to_csv(save_path, index=False)
