'''
This module contains functions to extract features from preprocessed MEP data,
to clean the feature set, to aggregate the features from MEP epoch level
to muscle level, to create dataframes containing features and outcome labels, 
and to perform feature selection. 

This is only used for the supervised learning model. 

Author: B.C. Burema
Date: 07/08/2026
'''

###-------IMPORTING LIBRARIES-------###
import os
import pandas as pd
import numpy as np
import tsfresh
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import RobustScaler
import gc


###------FEATURE EXTRACTION------###
def extract_features_tsfresh(ch, df, save_path):
    '''
    Extract features from MEP epochs using TSFRESH (https://tsfresh.readthedocs.io/en/latest/index.html).
    Save extracted features per muscle in csv file.

    Input:
        ch (str)            : Channel name.
        df (pd.Dataframe)   : Dataframe containing preprocessed MEPs of the specified muscle of all participants.
                              Columns: time, condition, epoch, value, sfreq, participant_id, value_scaled.
        save_path (Path)    : Path to folder in which the features are stored.
    '''

    # Create filename to save features in csv per muscle
    filename = f'features_{ch}.csv'
    save_loc = os.path.join(save_path, filename)

    # If csv file already exist, skip feature extraction
    if os.path.exists(save_loc):
        print(f'File with extracted features for {ch} already exists.')
        extracted_features = pd.read_csv(save_loc, sep=';')
    else:
    # Perform feature extraction
        extracted_features = tsfresh.extract_features(
                                df,
                                column_id='id', 
                                column_sort='time',
                                column_value='value_scaled'
                                )
        
        # Clean dataframe containing extracted features
        extracted_features.columns           = [col.replace('value_scaled__','') for col in extracted_features.columns]
        extracted_features                   = extracted_features.reset_index() # index becomes epoch_id
        extracted_features.rename(columns={'index': 'epoch_id'}, inplace=True)  # change column name to epoch_id
        extracted_features['participant_id'] = extracted_features['epoch_id'].str.split('_').str[0] # add participant_id column -> required for aggregation function
        extracted_features.to_csv(save_loc, index=False) 
    return None


###------FEATURE PREPROCESSING------###
def handle_missing_values(features_dict): 
    '''
    Remove features with any NaNs. We can be this strict, because there are a lot of features.

    Input:
        features_dict (dict): Dictionary containing dataframes with features per muscle.
    Output:
        features_dict (dict): Difference with input: Features with NaN values have been removed.
    '''

    threshold = 0 # you can adjust this threshold if you accept some NaNs
    
    # Loop through dataframes in dictionary
    for ch, features in features_dict.items():
        # Determine amount of NaNs in each column (feature)
        fraction_missing  = features.isnull().sum() / len(features) 
        
        # Only include feature when there are zero NaNs
        fraction_include  = fraction_missing == threshold #
        fraction_features = features.loc[:, fraction_include].copy()
        features_dict[ch] = fraction_features
    
    # Only include features present in all dataframes (all muscles)
    col_sets    = [set(features.columns) for features in features_dict.values()]
    common_cols = set.intersection(*col_sets)  

    for ch in features_dict:
        features_dict[ch] = features_dict[ch][list(common_cols)].copy()
    return features_dict
    
def remove_weird_features(features_dict): 
    '''
    Remove features that can hardly be explained to clinicians. 
    We want the features to be explainable, in order to potentially adjust
    the alarm criteria for MEP monitoring during surgery.

    Input:
        features_dict (dict): Dictionary containing dataframes with features per muscle.
    Output:
        features_dict (dict): Difference with input: Features that can hardly be explained, 
                              have been removed.
    '''
   
    exclude = ['agg_autocorrelation',
                'agg_linear_trend',
                'approximate_entropy',
                'autocorrelation__lag_0',                           # Remove this feature, because it's its correlation with exact itself...
                'cid_ce',
                'fft_aggregated',
                'fourier_entropy',
                'has_duplicate',
                'large_standard_deviation',
                'linear_trend',
                'partial_autocorrelation',
                'partial',
                'percentage_of_reoccurring_datapoints_to_all_datapoints',
                'percentage_of_reoccurring_datapoints_to_all_values',
                'percentage_of_reoccurring_values_to_all_values',
                'permutation_entropy',
                'ratio_value_number_to',
                'sample_entropy',
                'spkt_welch_density',
                'sum_of_reoccuring_datapoints',
                'sum_of_reoccurring_values',
                'variance_larger_than_standard_deviation',         
                'ar_coefficient',                                   # Above this feature, the features could be explained to clinicians, but these are already quite difficult. Below this feature, it is nearly impossible to explain.
                'augmented_dickey_fuller',
                'benford_correlation',
                'binned_entropy',
                'c3',
                'change_quantiles',
                'cwt_coefficients',
                'fft_coefficient',
                'friedrich_coefficients',
                'index_mass',
                'lempel_ziv_complexity',
                'linear_trend_timewise',
                'matrix_profile',
                'max_langevin_fixed_point',
                'mean_second_derivative_central',
                'number_cwt_peaks',
                'query_similarity',
                'set_property',
                'symmetry_looking',
                'time_reversal_asymmetry_statistic',
                ]

    # Check whether one of the above mentioned features are present in the feature name. If so: exclude.
    pattern = '|'.join(exclude) 

    for ch, features in features_dict.items():
        features_dict[ch] = features.loc[:, ~features.columns.str.contains(pattern)].copy() 
    return features_dict

def correlation_features(features_dict):
    '''
    Determine correlation between muscles, in order to determine whether features from
    individual muscles should be used or a combination should be created.
        Low correlation: use features from both individual muscles
        High correlation: choose features from one muscle, or make a combination.
    
    Input:
        features_dict (dict): Dictionary containing dataframes with features per muscle.
    '''
    corr_pairs = [
        ('TA_L', 'GAS_L'),      # Left foot
        ('AH_L', 'GAS_L'),      # Left foot
        ('AH_L', 'TA_L'),       # Left foot
        ('TA_R', 'GAS_R'),      # Right foot
        ('AH_R', 'GAS_R'),      # Right foot
        ('AH_R', 'TA_R'),       # Right foot
    ]

    # Define feature columns based on one muscle -> feature columns are the same for each muscle
    feature_cols = [col for col in features_dict['TA_L'].columns if col not in ['epoch_id','participant_id']] 
    
    corr_results = {}

    # Determmine correlation between muscle pairs
    for muscle_1, muscle_2 in corr_pairs:
        sorted_muscle_1     = features_dict[muscle_1].sort_values(
            by=['participant_id','epoch_id'],
            key=lambda x: x.str.split('_').str[1].astype(int) if x.name == 'epoch_id' else x)
        sorted_muscle_2     = features_dict[muscle_2].sort_values(
            by=['participant_id','epoch_id'],
            key=lambda x: x.str.split('_').str[1].astype(int) if x.name == 'epoch_id' else x)

        features_muscle_1   = sorted_muscle_1[feature_cols]
        features_muscle_2   = sorted_muscle_2[feature_cols]

        corr                = features_muscle_1.corrwith(features_muscle_2, method='spearman').abs()
        corr_results[f'{muscle_1}_vs_{muscle_2}'] = corr
    
    df_cross_corr                                 = pd.DataFrame(corr_results)

    # Visualize correlation
    df_cross_corr.abs().hist(bins=28, figsize=(14,8))
    plt.suptitle('Distribution correlation between muscles within extremity')
    plt.show()

    print(f'Mean correlations: {df_cross_corr.abs().mean()}')
    print(f'Number of features with correlation >0.7: {(df_cross_corr.abs()>0.7).sum()}')

    plt.figure(figsize=(10, len(df_cross_corr)*0.2))
    sns.heatmap(df_cross_corr, cmap='coolwarm',vmin=0,vmax=1, annot=False, yticklabels=True)
    plt.title(f'Heatmap of Spearman correlation between muscles within extremity')
    plt.xlabel(f'Muscle pairs')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
    return None

def clean_features(features_dict):
    '''
    Combine the above three functions to clean the features per muscle.

    Input:
        features_dict (dict)    : Dictionary containing dataframes with features per muscle.
    Output:
        features_removed (dict) : Difference from input: Features containing NaNs,
                                  and features that can hardly be explained to clinicians have
                                  been removed.
    '''
    # Handle missing values
    features_dict_nan = handle_missing_values(features_dict)
    # Remove features that can hardly be explained
    features_removed  = remove_weird_features(features_dict_nan)
    # Check correlation between muscles of extremity -> a plot is created
    correlation_features(features_removed)
    return features_removed


###------FEATURE AGGREGATION------###
### Feature aggregation per muscle from mep level to patient level
def aggregate_group(group, feature_cols, agg_functions):
    '''
    Aggregate features per muscle per participant. 

    Input:
        group (pd.Dataframe)  : Dataframe containing features of a specific muscle of one participant.
        feature_cols (list)   : List of feature names.
        agg_functions (dict)  : Dictionary containing aggregation functions.
    Output:
        pd.Series(aggregation): Series containing a value for each feature / aggregation combination.
    '''
    aggregation = {}
    for feature in feature_cols:
        for agg_name, agg_func in agg_functions.items():
            aggregation[f'{feature}__{agg_name}'] = agg_func(group[feature])
    return pd.Series(aggregation)

def aggregate_features(features_dict, agg_functions):
    '''
    Aggregate features per muscle per participant. We transform features from 
    epoch level to participant muscle level.

    Input:
        features_dict (dict) : Dictionary containing dataframes with features per muscle.
        agg_functions (dict) : Dictionary containing aggregation functions.
    Output:
        features_dict (dict) : Difference from input: Features are aggregated per muscle per participant.
    '''

    # Loop through feature dataframes
    for ch, features in features_dict.items():
        feature_cols = [col for col in features.columns if col not in ['epoch_id','participant_id']] 

        # Change epoch numbers to 001 and 012 instead of 1 and 12.
        features['epoch_id'] = features['epoch_id'].apply(
            lambda x: '_'.join(x.rsplit('_',1)[:-1]) + '_' + x.rsplit('_',1)[-1].zfill(3)
        )
        # Epochs should be sorted in the right order. Otherwise aggregation function max diff makes no sense,
        # since this calculates the difference between consecutive epochs.
        features = features.sort_values(
            by=['participant_id','epoch_id'],
            key=lambda x: x.str.split('_').str[1].astype(int) if x.name == 'epoch_id' else x) 
 
        # Aggregate features from MEP epoch level to participant muscle level
        feature_summary   = features.groupby('participant_id')[feature_cols].apply(lambda group: aggregate_group(group, feature_cols, agg_functions))
        del features
        gc.collect()

        feature_summary   = feature_summary.reset_index()

        features_dict[ch] = feature_summary
    return features_dict

### Create dataframes per extremity
def combine_feature_sets(features_dict, save_path, extremity, left_keys, right_keys, mapping, remove_participants):
    '''
    Create a dataframe per extremity. Combine features of QUAD and GAS for leg, and combine
    features of GAS, TA, and AH for foot.

    Input:
        features_dict (dict)      : Dictionary containing dataframes with features per muscle.
        save_path (str)           : Path to resultfolder.
        extremity (str)           : Either 'leg' or 'foot'.
        left_keys (list)          : List with muscle names of left side.
        right_keys (list)         : List with muscle names of right side.
        mapping (dictionary)      : Mapping to transform left en right keys to general muscle key.
        remove_participants (list): List with participant_ids that should be removed, 
                                    because not all muscles are present for these participants.
    '''

    subfolder     = 'tsfresh_features'
    filename      = f'features_{extremity}.csv'
    save_loc      = os.path.join(save_path, subfolder, filename)
    
    features_dict = {
        key: df[~df['participant_id'].isin(remove_participants)] for key, df in features_dict.items()
    }

    left_dfs     = []
    for key in left_keys:
        df       = features_dict[key]
        muscle   = mapping[key]
        df       = df.add_prefix(f'{muscle}__')
        df       = df.rename(columns={f'{muscle}__participant_id':'participant_id'})
        left_dfs.append(df)

    df_left      = left_dfs[0]
    for df in left_dfs[1:]:
        df_left  = df_left.merge(df, on='participant_id')
    df_left['side'] = 'L'

    right_dfs   = []
    for key in right_keys:
        df       = features_dict[key]
        muscle   = mapping[key]
        df       = df.add_prefix(f'{muscle}__')
        df       = df.rename(columns={f'{muscle}__participant_id':'participant_id'})
        right_dfs.append(df)

    df_right     = right_dfs[0]
    for df in right_dfs[1:]:
        df_right = df_right.merge(df, on='participant_id')
    df_right['side'] = 'R'

    df_right_left = pd.concat([df_left, df_right],axis=0).reset_index(drop=True)
    df_right_left.to_csv(save_loc, index=False) 
    return None 

### Create dataframe per extremity with outcomes
def combine_features_outcomes(features_bodypart, outcomes, left_col, right_col, save_loc, not_per_muscle=True):
    '''
    Add outcome label to features dataframe.

    Input:
        features_bodypart (pd.Dataframe): Dataframe containing features per extremity.
        outcomes (pd.Dataframe)         : Dataframe containing outcome labels per extremity per participant.
        left_col (str)                  : Column containing outcome label of left side.
        right_col (str)                 : Column containing outcome label of right side.
        save_loc (str)                  : Path to folder in which the dataframes are stored. 
        not_per_muscle (bool)           : Whether to merge from muscle to extremity level. 
                                          In our study we did not aggregate to extremity level.
    '''

    features_bodypart['participant_id'] = features_bodypart['participant_id'].astype(str)
    outcomes['participant_id'] = outcomes['participant_id'].astype(str)
    
    # Merge outcomes with feature dataframe
    df = features_bodypart.merge(
        outcomes[['participant_id',left_col,right_col]],
        on='participant_id'
    )

    df['outcome_label'] = df.apply(
        lambda row: row[left_col] if row['side'] == 'L' else row[right_col], axis=1
    )

    # Remove label of left and right side
    df = df.drop(columns=[left_col,right_col])

    if not_per_muscle: # this means you will aggregate to extremity level
        muscles = ['AH','TA','GAS','QUAD']
        aggregation_types = {'std', 'kurtosis', 'range','max_diff'}
        
        agg_direction = {}
        feature_names = set()

        for col in df.columns:
            for muscle in muscles:
                if col.startswith(muscle + '_'):
                    for agg in aggregation_types:
                        if col.endswith('_' + agg):
                            feature_name = col[len(muscle)+2:-len(agg)-2]
                            feature_names.add(feature_name)

        new_cols = {}
        for feat_name in feature_names:
            for agg in aggregation_types:
                cols = [f'{muscle}__{feat_name}__{agg}' for muscle in muscles if
                        f'{muscle}__{feat_name}__{agg}' in df.columns]
                if cols:
                    # aggregation to extremity level by using min and max 
                    new_cols[f'min__{feat_name}__{agg}'] = df[cols].min(axis=1)
                    new_cols[f'max__{feat_name}__{agg}'] = df[cols].max(axis=1)
        meta_cols = ['participant_id','outcome_label','side']
        extremity_df = pd.concat([df[meta_cols],pd.DataFrame(new_cols)],axis=1)
        extremity_df.to_csv(save_loc, index=False)
    else:
        df.to_csv(save_loc, index=False)
    return None


###------FEATURE SELECTION------###
### Feature selection before cross-validation
def pre_feature_selection(data, missing_threshold=0.1, var_threshold=0):
    '''
    Feature selection before cross-validation: remove features with >10% NaNs,
    check whether there are features with outliers, remove features with zero variance.
    
    Input:
        data (pd.Dataframe)    : Dataframe containing features of extremity. Either from leg or foot.
        missing_threshold (int): Threshold for accepted ratio missing values.
        var_threshold (int)    : Threshold for accepted variance.
    Output:
        X_var (pd.Dataframe)   : Dataframe containing features of extremity.
    '''

    X = data.copy()
    
    # Remove features with more than 10% NaNs (NaNs had arisen during aggregation)
    fraction_missing = X.isnull().sum() / len(X)
    fraction_include = fraction_missing <= missing_threshold 
    X_miss           = X.loc[:, fraction_include]

    print(f'Data -> missing data eruit: {data.shape[1]} -> {X_miss.shape[1]} features')
    
    # Impute remaining NaNs with mean of feature
    X_full = X_miss.transform(lambda x: x.fillna(x.mean()))
    
    # Determine fraction of outliers 
    for feat in X_full.columns:
        x             = X_full[feat]
        q1            = x.quantile(0.25)
        q3            = x.quantile(0.75)
        iqr           = q3-q1
        outliers      = ((x < q1 -1.5*iqr) | (x > q3 + 1.5*iqr)).sum()
        frac_outliers = outliers / len(x)

        print(f'For features {feat} fraction of outliers: {frac_outliers}')

    # Standardize features before calculating variance. Variance is dependent of scale of features.
    scaler   = RobustScaler() 
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_full),
        columns=X_full.columns,
        index=X_full.index
    )

    # Remove features with zero variance
    selector_var      = VarianceThreshold(threshold=var_threshold)
    selector_var.fit(X_scaled)
    selected_features = X_scaled.columns[selector_var.get_support()]
    X_var             = X_full.loc[:,selected_features] 

    print(f'Missing data -> zero var data eruit: {X_miss.shape[1]} -> {X_var.shape[1]} features')
    return X_var

### Feature selection within outer fold
def train_feature_selection(muscles, data, Y, anova_k, corr_threshold=0.9, not_per_muscle=False):
    '''
    Feature selection within outer fold.

    Input:
        muscles (list)          : List of muscle names.
        data (pd.Dataframe)     : Dataframe with features.
        Y (pd.Dataframe)        : Dataframe with outcome labels.
        anova_k (int)           : Anova k, indicating how many features can be selected.
        corr_threshold (int)    : Threshold for accepted correlation between features.
        not_per_muscle (bool)   : Whether to merge from muscle to extremity level. 
                                  In our study we did not aggregate to extremity level.
    Output:
        X_final (pd.Dataframe)  : Dataframe with selected features of train set.
        final_features (list)   : List of selected feature names -> used to select same set of features in test set.
    '''

    if not_per_muscle:
        f_scores, _ = f_classif(data, Y)
        anova_scores = pd.Series(f_scores, index=data.columns, name='f_score')

        # Determine which features are redundant using spearman correlation. 
        # One of the highly correlating features should be preserved; this is the one 
        # with the highest ANOVA score.
        spearman_corr_mat = data.corr(method='spearman').abs()
        upper_corr        = spearman_corr_mat.where(np.triu(np.ones(spearman_corr_mat.shape), k=1).astype(bool))

        to_drop           = set() 

        for col in upper_corr.columns: # col = current feature we're evaluating
            if col in to_drop:
                continue
            # Search for features that highly correlate with col -> create clusters of features
            corr_feat = upper_corr.index[upper_corr[col] > corr_threshold]

            for feat in corr_feat:
                if feat in to_drop:
                    continue
                # Compare ANOVA scores within clusters to determine which feature should be preserved
                if anova_scores[feat] < anova_scores[col]:
                    to_drop.add(feat)
                else:
                    to_drop.add(col)
        
        X_corr = data.drop(columns=list(to_drop))

        print(f'X var -> correlerende features eruit: {data.shape[1]} -> {X_corr.shape[1]} features')

        # Only preserve features with highest ANOVA score -> keep anova_k features
        selector_anova   = SelectKBest(score_func=f_classif, k=anova_k)
        X_anova          = selector_anova.fit_transform(X_corr, Y)
        X_anova          = pd.DataFrame(X_anova, columns=selector_anova.get_feature_names_out())

        print(f'X corr -> anova: {X_corr.shape[1]} -> {X_anova.shape[1]} features')

        final_features = X_anova.columns.tolist()
        X_final        = data[final_features]
        print(f'in total {X_final.shape[1]} features selected')
    else:
        # Equally partition anova k across muscle
        selected_per_muscle = []
        k_per_muscle        = max(1, anova_k//len(muscles))

        for muscle in muscles:
            muscle_feat = [feat for feat in data.columns if feat.startswith(muscle)]
            X_muscle    = data[muscle_feat]

            print(f'Aantal features {muscle}: {X_muscle.shape[1]}')
            
            # Calculate ANOVA score per feature = univariate testing
            f_scores, _  = f_classif(X_muscle, Y)
            anova_scores = pd.Series(f_scores, index=X_muscle.columns, name='f_score')

            # Determine which features are redundant using spearman correlation. 
            # One of the highly correlating features should be preserved; this is the one 
            # with the highest ANOVA score.
            spearman_corr_mat = X_muscle.corr(method='spearman').abs()
            upper_corr        = spearman_corr_mat.where(np.triu(np.ones(spearman_corr_mat.shape), k=1).astype(bool))

            to_drop           = set() 

            for col in upper_corr.columns: # col = current feature we're evaluating
                if col in to_drop:
                    continue
                # Search for features that highly correlate with col -> create clusters of features
                corr_feat = upper_corr.index[upper_corr[col] > corr_threshold]

                for feat in corr_feat:
                    if feat in to_drop:
                        continue
                    # Compare ANOVA scores within clusters to determine which feature should be preserved
                    if anova_scores[feat] < anova_scores[col]:
                        to_drop.add(feat)
                    else:
                        to_drop.add(col)
            
            X_corr = X_muscle.drop(columns=list(to_drop))

            print(f'X var -> correlerende features eruit: {X_muscle.shape[1]} -> {X_corr.shape[1]} features')

            # Only preserve features with highest ANOVA score -> keep anova_k features
            selector_anova   = SelectKBest(score_func=f_classif, k=k_per_muscle)
            X_anova          = selector_anova.fit_transform(X_corr, Y)
            X_anova          = pd.DataFrame(X_anova, columns=selector_anova.get_feature_names_out())

            print(f'X corr -> anova: {X_corr.shape[1]} -> {X_anova.shape[1]} features')

            selected_per_muscle.extend(X_anova.columns.tolist())
        
        X_final         = data[selected_per_muscle]
        print(f'in total {X_final.shape[1]} features selected')
        final_features  = selected_per_muscle
    return X_final, final_features
