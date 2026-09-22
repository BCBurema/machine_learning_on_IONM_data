'''
Main module to perform the complete supervised machine learning approach.
This is the only file that should be used. It can be run section by section.

Author: B.C. Burema
Date: 20/07/2026
'''

# %% Importing packages and modules
import os
import numpy as np
import pandas as pd
import gc
import pickle
import shap
from sklearn import metrics
from scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from TM3_Internship_BB.preprocessing.datautils import load_excel, load_data, create_df_per_ch, preprocessing
from TM3_Internship_BB.supervised_ml.datautils_sml import extract_features_tsfresh, clean_features, aggregate_features, combine_feature_sets, combine_features_outcomes, pre_feature_selection

from TM3_Internship_BB.supervised_ml.rf_classifier_functions import metrics_classifier
from TM3_Internship_BB.supervised_ml.rf_classifier import fit_model

from TM3_Internship_BB.supervised_ml.visualization_sml import single_roc, shap_scatter, visualize_cf_mat, ale_plot
from TM3_Internship_BB.visualization import plot_number_epochs

# %% 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width',None)

# %% Defining paths
data_path               = r'Z:\14_MMC_IONM_ML\3-Experiments\3-ConvData'
gosh_data_path          = os.path.join(data_path, r'Converted_GOSH_timestamp_sorted_modalities_decim')
metadata_path            = r'Z:\14_MMC_IONM_ML\3-Experiments\2-Data'
csv_path                = r'Z:\14_MMC_IONM_ML\3-Experiments\1-Inputs' 
EMC_csv_path            = csv_path + r'\SMART_export_20250217.csv'
GOSH_csv_path           = csv_path + r'\SMART_export_20260618.csv'
clinical_csv_paths      = [EMC_csv_path, GOSH_csv_path]
gosh_key_path           = csv_path + r'\SMART_GOSH_Key.xlsx'
resultfolder            = r'Z:\14_MMC_IONM_ML\6-Analysis_results\TM3_Internship_BB'
resultfolder_sml        = r'Z:\14_MMC_IONM_ML\6-Analysis_results\TM3_Internship_BB\results_supervised_learning'
preprocessed_data_path  = r'Z:\14_MMC_IONM_ML\6-Analysis_results\TM3_Internship_BB\preprocessed_data'
loaded_data_path        = r'Z:\14_MMC_IONM_ML\6-Analysis_results\TM3_Internship_BB\loaded_data'

# %% Loading data
deltamrc_threshold          = -1
timing                      = 'direct'
clinical_df, outcomes_mrc   = load_excel(clinical_csv_paths, gosh_key_path, deltamrc_threshold, timing)

# %% Print clinical data
remove_ids = ['EMC-029','EMC-033','EMC-034','EMC-036','EMC-037','EMC-040',
            'EMC-041','EMC-042','EMC-043','EMC-045','EMC-047','EMC-048','EMC-049',
            'SMART-019','SMART-119','SMART-150', 'SMART-151','SMART-152','SMART-153','SMART-154',
            'SMART-155','SMART-156','SMART-157','SMART-159','SMART-160','SMART-161',
            'SMART-161','SMART-163', 'SMART-164','SMART-165','SMART-167','SMART-168',
            'SMART-170','SMART-171','SMART-172']
clinical_df = clinical_df[~clinical_df['participant_id'].isin(remove_ids)]
gender      = clinical_df['pat_sex'].astype(str).str.strip()
n_female    = gender.isin(['Female','2']).sum()

print(f'Total number of patients: {len(clinical_df)}')
print(f'Median age at surgery: {clinical_df['age_at_surgery'].median()}')
print(f'Q1 age at surgery: {clinical_df['age_at_surgery'].quantile(0.25)}')
print(f'Q3 age at surgery: {clinical_df['age_at_surgery'].quantile(0.75)}')
print(f'Number of females: {n_female}')

# %% Create feature set
features_outcomes_loc       = os.path.join(resultfolder, 'tsfresh_features_outcomes')
features_loc                = os.path.join(resultfolder, 'tsfresh_features')

features_outcomes_leg_path  = os.path.join(features_outcomes_loc, 'features_outcomes_leg.csv')
features_outcomes_foot_path = os.path.join(features_outcomes_loc, 'features_outcomes_foot.csv')

features_leg_path           = os.path.join(features_loc, 'features_leg.csv')
features_foot_path          = os.path.join(features_loc, 'features_foot.csv')

muscle_keys                 = ['QUAD_L','QUAD_R',
                                'GAS_L','GAS_R',
                                'TA_L','TA_R',
                                'AH_L','AH_R']
muscle_paths                = [os.path.join(features_loc, f'features_{k}.csv') for k in muscle_keys]

# Initialize aggregation functions
agg_functions = {
    'range':        lambda x: np.percentile(x,95)-np.percentile(x,5),
    'std':          lambda x: np.std(x),
    'kurtosis':     lambda x: stats.kurtosis(x, fisher=True), #fisher = true-> normaal = 0 ipv 3
    'max_diff':     lambda x: np.percentile(np.abs(np.diff(x.values)), 95)
    }

while True:
    # Check whether leg and foot features_outcomes already exist -> load these dataframes 
    if os.path.exists(features_outcomes_leg_path) and os.path.exists(features_outcomes_foot_path):
        print(f'Load features_outcomes dataframes foot and leg')
        df_leg  = pd.read_csv(features_outcomes_leg_path, sep=',')
        df_foot = pd.read_csv(features_outcomes_foot_path, sep=',')
        break

    # Check whether leg and foot features already exist -> load these dataframes and combine with outcome labels
    elif os.path.exists(features_leg_path) and os.path.exists(features_foot_path):
        print('Load features dataframes foot and leg -> create df with features and outcomes per leg and foot')
        features_leg    = pd.read_csv(features_leg_path, sep=',')
        features_foot   = pd.read_csv(features_foot_path, sep=',')

        df_leg          = combine_features_outcomes(features_leg, outcomes_mrc, 'outcome_label_left_leg', 'outcome_label_right_leg', features_outcomes_leg_path, not_per_muscle=False)
        df_foot         = combine_features_outcomes(features_foot, outcomes_mrc, 'outcome_label_left_foot', 'outcome_label_right_foot', features_outcomes_foot_path, not_per_muscle=False)

    # Check whether features of all muscles already exist -> combine per extremity (create a dataframe for leg and foot)
    elif all(os.path.exists(p) for p in muscle_paths):
        print('Load features per muscle -> merge per leg and foot -> create df with features per leg and foot')
        features_per_ch = {}
        for file in os.listdir(features_loc):
            if not file.lower().endswith('.csv'):
                continue
            ch_name                     = file.replace('features_','').replace('.csv','')
            features_path               = os.path.join(features_loc, file)
            features_per_ch[ch_name]    = pd.read_csv(features_path)

        # Clean feature sets: remove zero variance, handle missing values, check correlation
        features_per_ch                 = clean_features(features_per_ch)
        
        # Aggregate from MEP level to muscle level per participant
        features_per_ch                 = aggregate_features(features_per_ch, agg_functions)
    
        # Combine features from different muscles in the same dataframe -> separate for leg and foot
        mapping_foot                    = {
                                            'GAS_L': 'GAS', 'GAS_R': 'GAS',
                                            'TA_L': 'TA', 'TA_R': 'TA',
                                            'AH_L': 'AH', 'AH_R': 'AH'
                                        }

        mapping_leg                     = {
                                            'QUAD_L': 'QUAD', 'QUAD_R': 'QUAD',
                                        }

        # Participant that should be removed from further analyses since these participants lack certain muscle(s) 
        remove_ids = ['EMC-029','EMC-033','EMC-034','EMC-036','EMC-037','EMC-040',
            'EMC-041','EMC-042','EMC-043','EMC-045','EMC-047','EMC-048','EMC-049',
            'SMART-019','SMART-119','SMART-150', 'SMART-151','SMART-152','SMART-153','SMART-154',
            'SMART-155','SMART-156','SMART-157','SMART-159','SMART-160','SMART-161',
            'SMART-161','SMART-163', 'SMART-164','SMART-165','SMART-167','SMART-168',
            'SMART-170','SMART-171','SMART-172']
        
        # Features leg
        combine_feature_sets(
            features_per_ch, 
            resultfolder, 
            extremity='leg',
            left_keys=['QUAD_L'],
            right_keys=['QUAD_R'],
            mapping=mapping_leg,
            remove_participants=remove_ids)

        # Features foot
        combine_feature_sets(
            features_per_ch, 
            resultfolder,
            extremity='foot',
            left_keys=['GAS_L','TA_L','AH_L'],
            right_keys=['GAS_R','TA_R','AH_R'],
            mapping=mapping_foot,
            remove_participants=remove_ids)

    # Check whether the preprocessed MEP dataframes already exist -> load these and start tsfresh feature extraction
    elif os.path.exists(os.path.join(preprocessed_data_path, 'preprocessed_MEP_TA_L.csv')):
        print('Preprocessed data found -> start tsfresh -> create df per muscle with features')
        # Extract TSFRESH features for each MEP from each muscle
        for file in os.listdir(preprocessed_data_path):
            if not file.lower().endswith('.csv'):
                continue

            ch_name                 = file.replace('preprocessed_MEP_','').replace('.csv','')
            print(f'Execute TSFRESH for {ch_name}')
            preprocessed_ch_path    = os.path.join(preprocessed_data_path, file)
            df                      = pd.read_csv(preprocessed_ch_path)
        
            # Extract featuers from preprocessed MEPs
            extract_features_tsfresh(ch_name, df, features_loc)

    # Check whether the raw MEP dataframes can be loaded -> perform preprocessing
    elif os.path.exists(os.path.join(loaded_data_path, 'meps.pkl')):
        print('Loaded data found -> start preprocessing -> create df per muscle with preprocesed meps')
        with open(os.path.join(loaded_data_path, 'meps.pkl'), 'rb') as f:
            meps = pickle.load(f)
        with open(os.path.join(loaded_data_path, 'participant_ids.pkl'), 'rb') as f:
            participant_ids = pickle.load(f)
        with open(os.path.join(loaded_data_path, 'start_idx_epochs.pkl'), 'rb') as f:
            start_idx_epochs = pickle.load(f)
        
        # Preprocess raw data of MEPs -> GEEFT EEN PREPROCESSED DF PER CH
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

    # If nothing exist yet -> start with loading data
    else:
        print('No features found, start loading meps') 
        # Load raw data of MEPs EMC
        meps_emc, participant_ids_emc, start_idx_epochs_emc, surgical_durations_emc = load_data(
            data_path=data_path, 
            metadata_path=metadata_path, 
            clinical_info=clinical_df, 
            file_pattern='*MEP-epo.fif', 
            id_pattern=r'(\d{3})_MEP', 
            institution='EMC')

        # Load raw data of MEPs GOSH
        meps_gosh, participant_ids_gosh, start_idx_epochs_gosh, surgical_durations_gosh= load_data(
            data_path=gosh_data_path, 
            metadata_path=metadata_path, 
            clinical_info=clinical_df, 
            file_pattern=None, 
            id_pattern=None,
            institution='GOSH')

        # Combine EMC and GOSH data
        meps                = meps_emc + meps_gosh
        participant_ids     = participant_ids_emc | participant_ids_gosh
        start_idx_epochs    = pd.concat([start_idx_epochs_emc, start_idx_epochs_gosh],ignore_index=True)
        plot_number_epochs(meps) 
        with open(os.path.join(loaded_data_path, 'meps.pkl'),'wb') as f:
            pickle.dump(meps, f)
        with open(os.path.join(loaded_data_path, 'participant_ids.pkl'),'wb') as f:
            pickle.dump(participant_ids, f)
        with open(os.path.join(loaded_data_path, 'start_idx_epochs.pkl'),'wb') as f:
            pickle.dump(start_idx_epochs, f)

# %% Define training data + training label
y_leg       = df_leg['outcome_label'].copy()
data_leg    = df_leg.copy().drop(columns=['participant_id','side','outcome_label'])

y_foot      = df_foot['outcome_label'].copy()
data_foot   = df_foot.copy().drop(columns=['participant_id','side','outcome_label'])

# Store participant ID for stratification
participant_ids_leg  = df_leg['participant_id'].copy()
participant_ids_foot = df_foot['participant_id'].copy()

# %% Pre feature selection 
selected_data_leg  = pre_feature_selection(data_leg)
selected_data_foot = pre_feature_selection(data_foot)

# %% Fit the model
# Initialize parameter grid for each type of classifier
rf_parameters = {
    "n_estimators":         list(range(200, 400)),
    "min_samples_split":    [10, 20, 30], 
    "min_samples_leaf":     [5, 10, 15, 20], 
    "criterion":            ["gini", "entropy"],
    "bootstrap":            [True, False],
    "max_depth":            [3, 5, 7, 9] 
}

brf_parameters = {
    "n_estimators":         list(range(200,400)),
    "min_samples_split":    [10, 20, 30], 
    "min_samples_leaf":     [5, 10, 15, 20],
    "criterion":            ["gini", "entropy"],
    "max_depth":            [3, 5, 7, 9], 
    "max_features":         ['sqrt','log2',0.3]
}


xgb_parameters = { 
    'n_estimators':         [100,200,300], 
    'max_depth':            [3, 5, 7, 9], 
    'learning_rate':        [0.01, 0.1, 0.3],
    'subsample':            [0.6, 0.8, 1.0], 
    'colsample_bytree':     [0.6, 0.8, 1.0], 
    'min_child_weight':     [5, 10, 15, 20]
}

# Combine the above defined grids into a dictionary 
param_grids = { 
    'RF':   rf_parameters,
    'BRF':  brf_parameters,
    'XGB':  xgb_parameters
}

# Perform randomized search with cross-validation for hyperparameter optimization
# Default score is accuracy. In this case, we want to optimize for recall because we want to reduce the amount of false negatives
# Based on https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65
scorers = {
    "precision_score":  metrics.make_scorer(metrics.precision_score),
    "recall_score":     metrics.make_scorer(metrics.recall_score),
    "accuracy_score":   metrics.make_scorer(metrics.accuracy_score),
    "f2_score":         metrics.make_scorer(metrics.fbeta_score, beta=2),
}
refit_score     = "recall_score"#"f2_score" 

# Define the muscles that should be analyzed per extremity 
prefixes_leg    = ['QUAD']
prefixes_foot   = ['GAS','TA','AH']

# Initialize values for training 
classifier      = 'brf'
threshold       = 0.4 
anova_k_leg     = 10
anova_k_foot    = 30
not_per_muscle  = False # False = per muscle, True = combination of muscles by aggregating features min and max 

print('Train BRF') # Choose between leg and foot:

# LEG
# metrics_per_fold, x_test_parts, x_test_full, y_true, y_chance, y_predict, mean_fpr, mean_tpr, tprs_lower, tprs_upper, mean_auc, sd_auc, list_shap_values, list_test_sets, list_expected_values, list_features, stable_features, ale_data = fit_model(selected_data_leg, prefixes_leg, y_leg, participant_ids_leg, classifier=classifier, n_splits=5, param_grids=param_grids,anova_k=anova_k_leg, threshold=threshold, not_per_muscle=not_per_muscle, scorers=scorers, refit_score=refit_score, resultfolder=resultfolder_sml+ r"\leg_model",plot=True)

# FOOT
metrics_per_fold, x_test_parts, x_test_full, y_true, y_chance, y_predict, mean_fpr, mean_tpr, tprs_lower, tprs_upper, mean_auc, sd_auc, list_shap_values, list_test_sets, list_expected_values, list_features, stable_features, ale_data = fit_model(selected_data_foot, prefixes_foot, y_foot, participant_ids_foot, classifier=classifier, n_splits=5, param_grids=param_grids,anova_k=anova_k_foot, threshold=threshold, not_per_muscle=not_per_muscle, scorers=scorers, refit_score=refit_score, resultfolder=resultfolder_sml+r"\foot_model",plot=True)

# %% Evaluate model performance 
# Choose between leg and foot
folder_foot_leg = r'\foot_model' # or r'\leg_model'

### Scores 
true_outcome    = pd.concat(y_true)
y_chance        = np.concatenate(y_chance)

det_curve(true_outcome, y_chance, resultfolder_sml+ folder_foot_leg)

fold_scores     = [metrics[1] for metrics in metrics_per_fold]
scores_mean     = pd.concat(fold_scores).groupby(level=0).mean()
scores_std      = pd.concat(fold_scores).groupby(level=0).std()

n_folds         = 5
t_crit          = stats.t.ppf(0.975, df=n_folds-1)

scores_ci       = scores_std* t_crit / np.sqrt(n_folds)

scores_report = pd.DataFrame({
    'mean':     scores_mean.loc['Value'],
    'std':      scores_std.loc['Value'],
    'ci_low':   (scores_mean.loc['Value'] - scores_ci.loc['Value']).clip(lower=0),
    'ci_high':  (scores_mean.loc['Value'] + scores_ci.loc['Value']).clip(upper=1)
})

print(metrics_per_fold)
print(scores_report)

### Plot single ROC - not used for thesis 
single_roc(resultfolder_sml+folder_foot_leg, mean_fpr, mean_tpr, tprs_lower, tprs_upper, mean_auc, sd_auc)

### Generate confusion matrix for all test sets combined
predicted_outcome   = np.concatenate(y_predict)
cf_mat              = metrics.confusion_matrix(true_outcome, predicted_outcome) 
cm_RF, scores_RF    = metrics_classifier(true_outcome, predicted_outcome,y_chance)      

visualize_cf_mat(cf_mat, resultfolder_sml+ folder_foot_leg)

print(scores_RF)

### Explainability - stable features
# ALE plot
ale_plot(stable_features, ale_data, resultfolder_sml+folder_foot_leg)

# Shap scatter plot
for feature in stable_features:
    feature_values  = []
    shap_values     = []
    outcomes        = []

    for shap_fold, x_fold, y_fold in zip(list_shap_values, x_test_parts, y_true):
        if feature not in x_fold.columns:
            continue # Skip fold if feature does not exist
        idx = x_fold.columns.get_loc(feature)
        feature_values.extend(x_fold[feature])
        shap_values.extend(shap_fold[:,idx])
        outcomes.extend(y_fold)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8,5))
    colors = ['#CF8126' if y==1 else '#5AA5DB' for y in outcomes]
    ax.scatter(feature_values, shap_values, c=colors,alpha=0.6)
    legend_elements = [
        Line2D([0],[0],marker='o',color='w',markerfacecolor='#CF8126',
        markersize=8, label='Deterioration'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor='#5AA5DB',
        markersize=8, label='Stable'),
    ]
    ax.axhline(y=0, color='black',linestyle='--',linewidth=0.8)
    ax.set_xlabel(f'{feature}', fontsize=14)
    ax.set_ylabel('SHAP value',fontsize=14)
    ax.tick_params(axis='x',labelsize=14)
    ax.tick_params(axis='y',labelsize=14)
    ax.legend(handles=legend_elements, title='Outcome label',fontsize=14, title_fontsize=14)
    plt.tight_layout()
    plt.savefig(resultfolder_sml +folder_foot_leg + f'/SHAP_Scatter_{feature}')
    plt.close()
