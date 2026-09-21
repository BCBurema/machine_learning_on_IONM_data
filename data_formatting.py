'''
Module to use as standalone BEFORE training TS2Vec.
The data is formatted to the correct format -> all kind of labels are saved.

Author: B.C. Burema
Date: 21/09/2026
'''

# %% Importing packages and modules
from TM3_Internship_BB.preprocessing.datautils import load_excel, load_data, create_df_per_ch, preprocessing
from TM3_Internship_BB.contrastive_ml.datautils_cml import formatting_ts2vec_input
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import sys

# %% Defining paths
data_path               = r'Z:\14_MMC_IONM_ML\3-Experiments\3-ConvData'
gosh_data_path          = os.path.join(data_path, r'Converted_GOSH_timestamp_sorted_modalities_decim')
metadata_path           = r'Z:\14_MMC_IONM_ML\3-Experiments\2-Data'
csv_path                = r'Z:\14_MMC_IONM_ML\3-Experiments\1-Inputs' 
EMC_csv_path            = csv_path + r'\SMART_export_20250217.csv'
GOSH_csv_path           = csv_path + r'\SMART_export_20260618.csv'
clinical_csv_paths      = [EMC_csv_path, GOSH_csv_path]
gosh_key_path           = csv_path + r'\SMART_GOSH_Key.xlsx'
resultfolder            = r'Z:\14_MMC_IONM_ML\6-Analysis_results\TM3_Internship_BB'
resultfolder_cl         = os.path.join(resultfolder, 'results_contrastive_learning')
figures_path            = os.path.join(resultfolder, 'participant_muscles')
data_encoder_path       = os.path.join(resultfolder, 'data_encoder')
preprocessed_data_path  = os.path.join(resultfolder, 'preprocessed_data')

# %% Loading data
deltamrc_threshold              = -1
timing                          = 'direct'
clinical_df, outcomes_mrc       = load_excel(clinical_csv_paths, gosh_key_path, deltamrc_threshold, timing)
_, _, _, surgical_durations_emc = load_data(data_path,metadata_path,clinical_df,'*MEP-epo.fif',r'(\d{3})_MEP','EMC')
_, _, _, surgical_durations_gosh = load_data(gosh_data_path, metadata_path, clinical_df, None, None, 'GOSH')
surgical_durations              = pd.concat([surgical_durations_emc, surgical_durations_gosh],ignore_index=True)

# %% Create input for TS2Vec
# Defining muscle groups per extremity
muscle_groups = {
    'left_leg': ['QUAD_L'],
    'right_leg': ['QUAD_R'],
    'left_foot': ['GAS_L','TA_L','AH_L'],
    'right_foot':['GAS_R','TA_R','AH_R']
}

# Defining paths to save the input for TS2Vec to
data_encoder_left_leg_path              = os.path.join(data_encoder_path, 'X_left_leg.npy')
data_encoder_right_leg_path             = os.path.join(data_encoder_path, 'X_right_leg.npy')
data_encoder_left_foot_path             = os.path.join(data_encoder_path, 'X_left_foot.npy')
data_encoder_right_foot_path            = os.path.join(data_encoder_path, 'X_right_foot.npy')

epoch_ids_left_leg_path                 = os.path.join(data_encoder_path, 'epoch_ids_left_leg.npy')
epoch_ids_right_leg_path                = os.path.join(data_encoder_path, 'epoch_ids_right_leg.npy')
epoch_ids_left_foot_path                = os.path.join(data_encoder_path, 'epoch_ids_left_foot.npy')
epoch_ids_right_foot_path               = os.path.join(data_encoder_path, 'epoch_ids_right_foot.npy')

epoch_patient_labels_left_leg_path      = os.path.join(data_encoder_path,  'epoch_patient_labels_left_leg.npy')
epoch_patient_labels_right_leg_path     = os.path.join(data_encoder_path,  'epoch_patient_labels_right_leg.npy')
epoch_patient_labels_left_foot_path     = os.path.join(data_encoder_path,  'epoch_patient_labels_left_foot.npy')
epoch_patient_labels_right_foot_path    = os.path.join(data_encoder_path,  'epoch_patient_labels_right_foot.npy')

# Define path depending on timing of outcome you're interested in
if timing == 'direct':
    epoch_outcome_labels_left_leg_path   = os.path.join(data_encoder_path, 'epoch_outcome_labels_left_leg.npy')
    epoch_outcome_labels_right_leg_path  = os.path.join(data_encoder_path, 'epoch_outcome_labels_right_leg.npy')
    epoch_outcome_labels_left_foot_path  = os.path.join(data_encoder_path, 'epoch_outcome_labels_left_foot.npy')
    epoch_outcome_labels_right_foot_path = os.path.join(data_encoder_path, 'epoch_outcome_labels_right_foot.npy')
elif timing == '3m':
    epoch_outcome_labels_left_leg_path   = os.path.join(data_encoder_path, 'epoch_outcome_labels_3m_left_leg.npy')
    epoch_outcome_labels_right_leg_path  = os.path.join(data_encoder_path, 'epoch_outcome_labels_3m_right_leg.npy')
    epoch_outcome_labels_left_foot_path  = os.path.join(data_encoder_path, 'epoch_outcome_labels_3m_left_foot.npy')
    epoch_outcome_labels_right_foot_path = os.path.join(data_encoder_path, 'epoch_outcome_labels_3m_right_foot.npy')
else:
    epoch_outcome_labels_left_leg_path   = os.path.join(data_encoder_path, 'epoch_outcome_labels_12m_left_leg.npy')
    epoch_outcome_labels_right_leg_path  = os.path.join(data_encoder_path, 'epoch_outcome_labels_12m_right_leg.npy')
    epoch_outcome_labels_left_foot_path  = os.path.join(data_encoder_path, 'epoch_outcome_labels_12m_left_foot.npy')
    epoch_outcome_labels_right_foot_path = os.path.join(data_encoder_path, 'epoch_outcome_labels_12m_right_foot.npy')

while True:
    # Load data if these already exist
    if os.path.exists(data_encoder_left_leg_path) and os.path.exists(data_encoder_right_leg_path) and os.path.exists(data_encoder_left_foot_path) and os.path.exists(data_encoder_right_foot_path):
        print(f'Load data of both feet and legs for TS2Vec')
        X_left_leg                      = np.load(data_encoder_left_leg_path)
        left_leg_epoch_patient_labels   = np.load(epoch_patient_labels_left_leg_path, allow_pickle=True)
        left_leg_epoch_outcome_labels   = np.load(epoch_outcome_labels_left_leg_path, allow_pickle=True)
        left_leg_ids                    = np.load(epoch_ids_left_leg_path, allow_pickle=True).tolist()

        X_right_leg                     = np.load(data_encoder_right_leg_path)
        right_leg_epoch_patient_labels  = np.load(epoch_patient_labels_right_leg_path, allow_pickle=True)
        right_leg_epoch_outcome_labels  = np.load(epoch_outcome_labels_right_leg_path, allow_pickle=True)
        right_leg_ids                   = np.load(epoch_ids_right_leg_path, allow_pickle=True).tolist()

        X_left_foot                     = np.load(data_encoder_left_foot_path)
        left_foot_epoch_patient_labels  = np.load(epoch_patient_labels_left_foot_path, allow_pickle=True)
        left_foot_epoch_outcome_labels  = np.load(epoch_outcome_labels_left_foot_path, allow_pickle=True)
        left_foot_ids                   = np.load(epoch_ids_left_foot_path, allow_pickle=True).tolist()

        X_right_foot                    = np.load(data_encoder_right_foot_path)
        right_foot_epoch_patient_labels = np.load(epoch_patient_labels_right_foot_path, allow_pickle=True)
        right_foot_epoch_outcome_labels = np.load(epoch_outcome_labels_right_foot_path, allow_pickle=True)
        right_foot_ids                  = np.load(epoch_ids_right_foot_path, allow_pickle=True).tolist()
        break
    
    elif os.path.exists(os.path.join(preprocessed_data_path, 'preprocessed_MEP_AH_L.csv')):
        # If preprocessed data exists -> start formatting TS2Vec per extremity
        print('Preprocessed data found -> start formatting for TS2Vec')
        print('Start formatting left leg')
        left_leg_X, left_leg_ids, left_leg_epoch_patient_labels, left_leg_epoch_outcome_labels = formatting_ts2vec_input(
            data_encoder_path,
            'left_leg',
            preprocessed_data_path,
            outcomes_mrc,
            muscle_groups['left_leg'],
            timing=timing
        )  
        print(f'shape left leg = {left_leg_X.shape}')

        print('Start formatting right leg')
        right_leg_X, right_leg_ids, right_leg_epoch_patient_labels, right_leg_epoch_outcome_labels = formatting_ts2vec_input(
            data_encoder_path,
            'right_leg',
            preprocessed_data_path,
            outcomes_mrc,
            muscle_groups['right_leg'],
            timing=timing
        )
        print(f'shape right leg = {right_leg_X.shape}')

        print('Start formatting left foot')
        left_foot_X, left_foot_ids, left_foot_epoch_patient_labels, left_foot_epoch_outcome_labels = formatting_ts2vec_input(
            data_encoder_path,
            'left_foot',
            preprocessed_data_path,
            outcomes_mrc,
            muscle_groups['left_foot'],
            timing=timing
        )
        print(f'shape left foot = {left_foot_X.shape}')

        print('Start formatting right foot')
        right_foot_X, right_foot_ids, right_foot_epoch_patient_labels, right_foot_epoch_outcome_labels = formatting_ts2vec_input(
            data_encoder_path,
            'right_foot',
            preprocessed_data_path,
            outcomes_mrc,
            muscle_groups['right_foot'],
            timing=timing
        )
        print(f'shape right foot = {right_foot_X.shape}')

    elif os.path.exists(os.path.join(loaded_data_path, 'meps.pkl')):
        # If no preprocessed data exists -> load raw data and preprocess data
        print('Loaded data found -> start preprocessing -> create df per muscle with preprocesed meps')
        with open(os.path.join(loaded_data_path, 'meps.pkl'), 'rb') as f:
            meps             = pickle.load(f)
        with open(os.path.join(loaded_data_path, 'participant_ids.pkl'), 'rb') as f:
            participant_ids  = pickle.load(f)
        with open(os.path.join(loaded_data_path, 'start_idx_epochs.pkl'), 'rb') as f:
            start_idx_epochs = pickle.load(f)
        
        # Preprocess raw data of MEPs -> Returns dataframe containing preprocessed MEPs per channel
        create_df_per_ch(meps, start_idx_epochs, loaded_data_path) 
        
        for file in os.listdir(loaded_data_path):
            if not file.endswith('.csv'):
                continue
            ch_name     = file.replace('MEP_','').replace('.csv','')
            mep_path    = os.path.join(loaded_data_path, file)
            df          = pd.read_csv(mep_path)

            preprocessing(ch_name, df, preprocessed_data_path)
            del df
            gc.collect()
    else:
        print('No data for TS2Vec or preprocessed data found, start loading meps')
        # Load raw data of MEPs EMC
        meps_emc, participant_ids_emc, start_idx_epochs_emc = load_data(
            data_path=data_path, 
            metadata_path=metadata_path, 
            clinical_info=clinical_df, 
            file_pattern='*MEP-epo.fif', 
            id_pattern=r'(\d{3})_MEP', 
            institution='EMC')

        # Load raw data of MEPs GOSH
        meps_gosh, participant_ids_gosh, start_idx_epochs_gosh = load_data(
            data_path=gosh_data_path, 
            metadata_path=metadata_path, 
            clinical_info=clinical_df, 
            file_pattern=None, 
            id_pattern=None,
            institution='GOSH')

        # Combine EMC and GOSH data
        meps             = meps_emc + meps_gosh
        participant_ids  = participant_ids_emc | participant_ids_gosh
        start_idx_epochs = pd.concat([start_idx_epochs_emc, start_idx_epochs_gosh],ignore_index=True)
        plot_number_epochs(meps)

        with open(os.path.join(loaded_data_path, 'meps.pkl'),'wb') as f:
            pickle.dump(meps, f)
        with open(os.path.join(loaded_data_path, 'participant_ids.pkl'),'wb') as f:
            pickle.dump(participant_ids, f)
        with open(os.path.join(loaded_data_path, 'start_idx_epochs.pkl'),'wb') as f:
            pickle.dump(start_idx_epochs, f)

# %% Perform padding such that both feet and legs have the same number of timesteps (manually assessed)
pad_width_foot  = ((0,0), (0,10),(0,0)) 
X_left_foot     = np.pad(X_left_foot, pad_width=pad_width_foot, mode='constant',constant_values=np.nan)

pad_width_leg   = ((0,0), (0,10),(0,0)) 
X_left_leg      = np.pad(X_left_leg, pad_width=pad_width_leg, mode='constant',constant_values=np.nan)

# %% Combine left and right side (also for all labels)
data_encoder_leg_path           = os.path.join(data_encoder_path, 'X_leg.npy')
data_encoder_foot_path          = os.path.join(data_encoder_path, 'X_foot.npy')

epoch_ids_leg_path              = os.path.join(data_encoder_path, 'epoch_ids_leg.npy')
epoch_ids_foot_path             = os.path.join(data_encoder_path, 'epoch_ids_foot.npy')

epoch_patient_labels_leg_path   = os.path.join(data_encoder_path,  'epoch_patient_labels_leg.npy')
epoch_patient_labels_foot_path  = os.path.join(data_encoder_path,  'epoch_patient_labels_foot.npy')

epoch_outcome_labels_leg_path   = os.path.join(data_encoder_path,  'epoch_outcome_labels_leg.npy')
epoch_outcome_labels_foot_path  = os.path.join(data_encoder_path,  'epoch_outcome_labels_foot.npy')

epoch_side_leg_path             = os.path.join(data_encoder_path, 'epoch_side_labels_leg.npy')
epoch_side_foot_path            = os.path.join(data_encoder_path, 'epoch_side_labels_foot.npy')

epoch_center_leg_path           = os.path.join(data_encoder_path, 'epoch_center_labels_leg.npy')
epoch_center_foot_path          = os.path.join(data_encoder_path, 'epoch_center_labels_foot.npy')

epoch_duration_leg_path         = os.path.join(data_encoder_path, 'epoch_surgical_duration_leg.npy')
epoch_duration_foot_path        = os.path.join(data_encoder_path, 'epoch_surgical_duration_foot.npy')

left_leg_ids = [
    '_'.join([parts[0],parts[1].zfill(3)])
    if len(parts) >1 else e
    for e in left_leg_ids
    for parts in [e.split('_')]
]

right_leg_ids = [
    '_'.join([parts[0],parts[1].zfill(3)])
    if len(parts) >1 else e
    for e in right_leg_ids
    for parts in [e.split('_')]
]

left_foot_ids = [
    '_'.join([parts[0],parts[1].zfill(3)])
    if len(parts) >1 else e
    for e in left_foot_ids
    for parts in [e.split('_')]
]

right_foot_ids = [
    '_'.join([parts[0],parts[1].zfill(3)])
    if len(parts) >1 else e
    for e in right_foot_ids
    for parts in [e.split('_')]
]

# Combine left and right sides (also for all labels)
epoch_ids_leg               = left_leg_ids + right_leg_ids
epoch_ids_foot              = left_foot_ids + right_foot_ids

side_labels_leg             = np.array(['left']*len(left_leg_ids) + ['right']*len(right_leg_ids))
side_labels_foot            = np.array(['left']*len(left_foot_ids) + ['right']*len(right_foot_ids))

X_leg                       = np.concatenate((X_left_leg, X_right_leg), axis=0)
X_foot                      = np.concatenate((X_left_foot, X_right_foot), axis=0)

epoch_patient_labels_leg    = np.concatenate((left_leg_epoch_patient_labels, right_leg_epoch_patient_labels))
epoch_patient_labels_foot   = np.concatenate((left_foot_epoch_patient_labels, right_foot_epoch_patient_labels))

epoch_outcome_labels_leg    = np.concatenate((left_leg_epoch_outcome_labels, right_leg_epoch_outcome_labels))
epoch_outcome_labels_foot   = np.concatenate((left_foot_epoch_outcome_labels, right_foot_epoch_outcome_labels))

# Sort on time
sort_idx                    = np.argsort(epoch_ids_leg)
epoch_ids_leg               = [epoch_ids_leg[i] for i in sort_idx]
X_leg                       = X_leg[sort_idx]
epoch_patient_labels_leg    = epoch_patient_labels_leg[sort_idx]
epoch_outcome_labels_leg    = epoch_outcome_labels_leg[sort_idx]
side_labels_leg             = side_labels_leg[sort_idx]

sort_idx                    = np.argsort(epoch_ids_foot)
epoch_ids_foot              = [epoch_ids_foot[i] for i in sort_idx]
X_foot                      = X_foot[sort_idx]
epoch_patient_labels_foot   = epoch_patient_labels_foot[sort_idx]
epoch_outcome_labels_foot   = epoch_outcome_labels_foot[sort_idx]
side_labels_foot            = side_labels_foot[sort_idx]

# Create center and surgical duration labels
center_labels_leg            = np.where(np.char.find(epoch_patient_labels_leg.astype(str), 'EMC') >=0, 'EMC', 'GOSH')
center_labels_foot           = np.where(np.char.find(epoch_patient_labels_foot.astype(str), 'EMC') >=0, 'EMC', 'GOSH')

surgical_duration_map        = surgical_durations.set_index('participant_id')['surgical_duration'].to_dict()
epoch_surgical_duration_leg  = np.array([surgical_duration_map[pid] for pid in epoch_patient_labels_leg])
epoch_surgical_duration_foot = np.array([surgical_duration_map[pid] for pid in epoch_patient_labels_foot])

# Exclude patients
exclude_patients = ['EMC-029','EMC-033','EMC-034','EMC-036','EMC-037','EMC-040',
    'EMC-041','EMC-042','EMC-043','EMC-045','EMC-047','EMC-048','EMC-049',
    'SMART-019','SMART-119','SMART-159','SMART-160','SMART-161','SMART-161','SMART-163',
    'SMART-164','SMART-165','SMART-167','SMART-168','SMART-170','SMART-171',
    'SMART-172', 'SMART-150', 'SMART-151','SMART-152','SMART-153','SMART-154',
    'SMART-155','SMART-156','SMART-157']

# foot
mask_foot                   = ~np.isin(epoch_patient_labels_foot, exclude_patients)
X_foot                      = X_foot[mask_foot]
epoch_ids_foot              = np.array(epoch_ids_foot)[mask_foot]
epoch_patient_labels_foot   = epoch_patient_labels_foot[mask_foot]
epoch_outcome_labels_foot   = epoch_outcome_labels_foot[mask_foot]
side_labels_foot            = side_labels_foot[mask_foot]
center_labels_foot          = center_labels_foot[mask_foot]

print(f'foot unique patient labels: {len(np.unique(epoch_patient_labels_foot))}')
print(f'number of patient labels: {len(epoch_patient_labels_foot)}')
print(f'shape foot: {X_foot.shape}')

# leg
mask_leg                    = ~np.isin(epoch_patient_labels_leg, exclude_patients)
X_leg                       = X_leg[mask_leg]
epoch_ids_leg               = np.array(epoch_ids_leg)[mask_leg]
epoch_patient_labels_leg    = epoch_patient_labels_leg[mask_leg]
epoch_outcome_labels_leg    = epoch_outcome_labels_leg[mask_leg]
side_labels_leg             = side_labels_leg[mask_leg]
center_labels_leg           = center_labels_leg[mask_leg]

print(f'leg unique patient labels: {len(np.unique(epoch_patient_labels_leg))}')
print(f'number of patient labels: {len(epoch_patient_labels_leg)}')
print(f'shape leg: {X_leg.shape}')

# Save all data
np.save(data_encoder_leg_path, X_leg)
np.save(data_encoder_foot_path, X_foot)
np.save(epoch_ids_leg_path, np.array(epoch_ids_leg))
np.save(epoch_ids_foot_path, np.array(epoch_ids_foot))
np.save(epoch_patient_labels_leg_path, epoch_patient_labels_leg)
np.save(epoch_patient_labels_foot_path, epoch_patient_labels_foot)
np.save(epoch_outcome_labels_leg_path, epoch_outcome_labels_leg)
np.save(epoch_outcome_labels_foot_path, epoch_outcome_labels_foot)
np.save(epoch_side_leg_path, side_labels_leg)
np.save(epoch_side_foot_path, side_labels_foot)
np.save(epoch_center_leg_path, center_labels_leg)
np.save(epoch_center_foot_path, center_labels_foot)
np.save(epoch_duration_leg_path, epoch_surgical_duration_leg)
np.save(epoch_duration_foot_path, epoch_surgical_duration_foot)


#%% Create labels for features of legs
# paths with TSFRESH features of quadriceps
quad_l_path         = os.path.join(resultfolder, 'tsfresh_features/features_QUAD_L.csv')
quad_r_path         = os.path.join(resultfolder, 'tsfresh_features/features_QUAD_R.csv')

# Paths of data in correct shape for TS2VEc
epoch_ids_leg_path      = os.path.join(data_encoder_path, 'epoch_ids_leg.npy')
epoch_ids_foot_path     = os.path.join(data_encoder_path, 'epoch_ids_foot.npy')
epoch_side_leg_path     = os.path.join(data_encoder_path, 'epoch_side_labels_leg.npy')
epoch_side_foot_path    = os.path.join(data_encoder_path, 'epoch_side_labels_foot.npy')
abs_energy_leg_path     = os.path.join(data_encoder_path, 'abs_energy_leg.npy')
abs_max_leg_path        = os.path.join(data_encoder_path, 'abs_max_leg.npy')
amplitude_leg_path      = os.path.join(data_encoder_path, 'amplitude_leg.npy')
latency_leg_path        = os.path.join(data_encoder_path, 'latency_leg.npy')
energy_ratio_6_path     = os.path.join(data_encoder_path, 'energy_ratio_focus_6.npy')
energy_ratio_7_path     = os.path.join(data_encoder_path, 'energy_ratio_focus_7.npy')
energy_ratio_8_path     = os.path.join(data_encoder_path, 'energy_ratio_focus_8.npy')
energy_ratio_9_path     = os.path.join(data_encoder_path, 'energy_ratio_focus_9.npy')
length_path             = os.path.join(data_encoder_path, 'length_leg.npy')

# Retrieve all epoch ids
epoch_ids_leg           = np.load(epoch_ids_leg_path)
epoch_sides_leg         = np.load(epoch_side_leg_path)

# Retrieve dataframes containing features per epoch
features_quad_l         = pd.read_csv(quad_l_path)
features_quad_r         = pd.read_csv(quad_r_path)

def pad_epoch_id(epoch_id):
    parts = str(epoch_id).split('_')
    if len(parts)==2:
        return(f'{parts[0]}_{parts[1].zfill(3)}')
    return epoch_id

features_quad_l['epoch_id'] = features_quad_l['epoch_id'].apply(pad_epoch_id)
features_quad_r['epoch_id'] = features_quad_r['epoch_id'].apply(pad_epoch_id)

# Look for epoch id in features dataframes
lookup_l = features_quad_l.set_index('epoch_id').to_dict('index')
lookup_r = features_quad_r.set_index('epoch_id').to_dict('index')

# Initialize lists per feature
abs_energy_list     = []
abs_max_list        = []
amplitude_list      = []
first_loc_max_list  = []
energy_ratio_6_list = []
energy_ratio_7_list = []
energy_ratio_8_list = []
energy_ratio_9_list = []
length_list         = []


for epoch_id, side in zip(epoch_ids_leg, epoch_sides_leg):
    lookup  = lookup_l if side == 'left' else lookup_r
    row     = lookup.get(epoch_id,None)

    if row is None:
        print(f'Not found: {epoch_id}, {side}')
        abs_energy_list.append(np.nan)
        abs_max_list.append(np.nan)
        amplitude_list.append(np.nan)
        first_loc_max_list.append(np.nan)

        energy_ratio_6_list.append(np.nan)
        energy_ratio_7_list.append(np.nan)
        energy_ratio_8_list.append(np.nan)
        energy_ratio_9_list.append(np.nan)

        length_list.append(np.nan)
    else:
        abs_energy_list.append(row['abs_energy'])
        abs_max_list.append(row['absolute_maximum'])
        amplitude_list.append(row['maximum']-row['minimum'])
        first_loc_max_list.append(row['first_location_of_maximum'])

        energy_ratio_6_list.append(row['energy_ratio_by_chunks__num_segments_10__segment_focus_6'])
        energy_ratio_7_list.append(row['energy_ratio_by_chunks__num_segments_10__segment_focus_7'])
        energy_ratio_8_list.append(row['energy_ratio_by_chunks__num_segments_10__segment_focus_8'])
        energy_ratio_9_list.append(row['energy_ratio_by_chunks__num_segments_10__segment_focus_9'])

        length_list.append(row['length'])

abs_energy          = np.array(abs_energy_list)
abs_max             = np.array(abs_max_list)
amplitude           = np.array(amplitude_list)
first_loc_maximum   = np.array(first_loc_max_list)

energy_ratio_6      = np.array(energy_ratio_6_list)
energy_ratio_7      = np.array(energy_ratio_7_list)
energy_ratio_8      = np.array(energy_ratio_8_list)
energy_ratio_9      = np.array(energy_ratio_9_list)

length              = np.array(length_list)

# Save features in correct data format for labelling
np.save(abs_energy_leg_path, abs_energy)
np.save(abs_max_leg_path, abs_max)
np.save(amplitude_leg_path,amplitude)
np.save(latency_leg_path, first_loc_maximum)

np.save(energy_ratio_6_path, energy_ratio_6)
np.save(energy_ratio_7_path, energy_ratio_7)
np.save(energy_ratio_8_path, energy_ratio_8)
np.save(energy_ratio_9_path, energy_ratio_9)

np.save(length_path, length)

# %% Create feature data for kurtosis -> all epochs from one extremity are assigned the same label
# Retrieve features dataframe
features_path                   = os.path.join(resultfolder, 'tsfresh_features_outcomes/features_outcomes_leg.csv')
features_df                     = pd.read_csv(features_path)
side_map                        = {'L':'left','R':'right'}
features_df['side']             = features_df['side'].map(side_map)

# Retrieve all data encoder paths
epoch_patient_labels_leg_path   = os.path.join(data_encoder_path,  'epoch_patient_labels_leg.npy')
epoch_side_leg_path             = os.path.join(data_encoder_path, 'epoch_side_labels_leg.npy')

side_labels_leg                 = np.load(epoch_side_leg_path)
epoch_patient_labels_leg        = np.load(epoch_patient_labels_leg_path)

kurt_energy_ratio_6_path        = os.path.join(data_encoder_path, 'kurt_energy_ratio_focus_6.npy')
kurt_energy_ratio_7_path        = os.path.join(data_encoder_path, 'kurt_energy_ratio_focus_7.npy')
kurt_energy_ratio_8_path        = os.path.join(data_encoder_path, 'kurt_energy_ratio_focus_8.npy')
kurt_energy_ratio_9_path        = os.path.join(data_encoder_path, 'kurt_energy_ratio_focus_9.npy')

kurtosis_map_6                  = features_df.set_index(['participant_id','side'])['QUAD__energy_ratio_by_chunks__num_segments_10__segment_focus_6__kurtosis'].to_dict()
kurtosis_map_7                  = features_df.set_index(['participant_id','side'])['QUAD__energy_ratio_by_chunks__num_segments_10__segment_focus_7__kurtosis'].to_dict()
kurtosis_map_8                  = features_df.set_index(['participant_id','side'])['QUAD__energy_ratio_by_chunks__num_segments_10__segment_focus_8__kurtosis'].to_dict()
kurtosis_map_9                  = features_df.set_index(['participant_id','side'])['QUAD__energy_ratio_by_chunks__num_segments_10__segment_focus_9__kurtosis'].to_dict()

# Retrieve kurtosis of energy ... per focus segment
epoch_kurtosis_6                = np.array([kurtosis_map_6[(pid,side)] for pid, side in zip (epoch_patient_labels_leg,side_labels_leg)])
epoch_kurtosis_7                = np.array([kurtosis_map_7[(pid,side)] for pid, side in zip (epoch_patient_labels_leg,side_labels_leg)])
epoch_kurtosis_8                = np.array([kurtosis_map_8[(pid,side)] for pid, side in zip (epoch_patient_labels_leg,side_labels_leg)])
epoch_kurtosis_9                = np.array([kurtosis_map_9[(pid,side)] for pid, side in zip (epoch_patient_labels_leg,side_labels_leg)])

# Save features in correct data format for labelling
np.save(kurt_energy_ratio_6_path, epoch_kurtosis_6)
np.save(kurt_energy_ratio_7_path, epoch_kurtosis_7)
np.save(kurt_energy_ratio_8_path, epoch_kurtosis_8)
np.save(kurt_energy_ratio_9_path,epoch_kurtosis_9)

# %% Create data for surgical duration -> all epochs from one extremity are assigned the same label
# Retrieve all data encoder paths
epoch_patient_labels_leg_path   = os.path.join(data_encoder_path,  'epoch_patient_labels_leg.npy')
epoch_patient_labels_foot_path  = os.path.join(data_encoder_path,  'epoch_patient_labels_foot.npy')

epoch_outcome_labels_leg_path   = os.path.join(data_encoder_path,  'epoch_outcome_labels_leg.npy')
epoch_outcome_labels_foot_path  = os.path.join(data_encoder_path,  'epoch_outcome_labels_foot.npy')

epoch_side_leg_path             = os.path.join(data_encoder_path, 'epoch_side_labels_leg.npy')
epoch_side_foot_path            = os.path.join(data_encoder_path, 'epoch_side_labels_foot.npy')

epoch_center_leg_path           = os.path.join(data_encoder_path, 'epoch_center_labels_leg.npy')
epoch_center_foot_path          = os.path.join(data_encoder_path, 'epoch_center_labels_foot.npy')

epoch_duration_leg_path         = os.path.join(data_encoder_path, 'epoch_surgical_duration_leg.npy')
epoch_duration_foot_path        = os.path.join(data_encoder_path, 'epoch_surgical_duration_foot.npy')

epoch_patient_labels_leg        = np.load(epoch_patient_labels_leg_path)
epoch_patient_labels_foot       = np.load(epoch_patient_labels_foot_path)

# Get surgical duration per participant id
surgical_durations['surgical_duration'] = surgical_durations['surgical_duration'].dt.total_seconds()/60
surgical_duration_map                   = surgical_durations.set_index('participant_id')['surgical_duration'].to_dict()
# Create arrays filled with surgical duration
epoch_surgical_duration_leg             = np.array([surgical_duration_map[pid] for pid in epoch_patient_labels_leg])
epoch_surgical_duration_foot            = np.array([surgical_duration_map[pid] for pid in epoch_patient_labels_foot])

# Save surgical durations in correct data format for labelling
np.save(epoch_duration_leg_path, epoch_surgical_duration_leg)
np.save(epoch_duration_foot_path, epoch_surgical_duration_foot)
