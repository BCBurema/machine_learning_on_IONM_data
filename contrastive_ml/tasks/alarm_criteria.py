'''
This module contains functions to calculate alarm criteria based on the learned representation space.

Author: B.C. Burema
Date: 21/09/2026
'''

###-------IMPORTING LIBRARIES------###
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from scipy import stats
from statsmodels.stats.multitest import multipletests

###-------FUNCTIONS TO COMPUTE ALARM CRITERIA------###
def normalized_auc(y):
    '''
    First normalize x-axis to [0,1] making sure that the number of MEPs per patient does not influence the summary alarm metric.
    Second, compute AUC of alarm metrics throughout surgery.
    
    Input:
        y (np.array): Array containing alarm metric values for each MEP 
    
    Output:
        auc (float) : AUC of alarm metric for one extremity
    '''

    y            = np.asarray(y)
    y            = y[~np.isnan(y)]
    n            =len(y)
    x_normalized = np.linspace(0,1,n)
    auc          = np.trapz(y,x_normalized)
    return auc

def compute_alarm_features(representations, patient_labels, side_labels, outcome_labels, epoch_ids):
    '''
    Compute alarm values per extremity.

    Input:
        representations :  Learned 320-dimensional representations of each MEP
        patient_labels  :  Patient labels assigned to each MEP
        side_labels     :  Side labels assigned to each MEP
        outcome_labels  :  Outcome labels assigned to each MEP
        epoch_ids       :  Epoch id per MEP
    Output:
        alarm_features  : Dataframe containing the alarm features per extremity
    '''

    alarm_features_extremity = []

    # Retrieve unique extremities by searching on the combination patient label and side label
    extremities = set(zip(patient_labels, side_labels))

    for patient, side in extremities:
        mask            = (patient_labels == patient) & (side_labels == side)
        repr_extremity  = representations[mask] 
        outcome         = int(outcome_labels[mask][0]) 
        mep_ids         = epoch_ids[mask]
        n_meps          = int(len(repr_extremity))

        if n_meps < 2:
            print(f'Skipping {patient} {side}: too few MEPs ({n_meps})')
            continue

        baseline = repr_extremity[0] # Take the first MEP as baseline

        # Alarm feature 1: Euclidean distance relative to baseline
        baseline_distances_euclidean = np.array([
            np.linalg.norm(repr_extremity[i]-baseline)
            for i in range(n_meps)
        ])

        # Alarm feature 2: Manhattan distance relative to baseline
        baseline_distances_manhattan = np.array([
            np.sum(np.abs(repr_extremity[i]-baseline))
            for i in range(n_meps)
        ])

        # Alarm feature 3: dot product relative to baseline
        baseline_dot = np.array([
            np.dot(repr_extremity[i],baseline) # higher value = greater similarity
            for i in range(n_meps)
        ])

        # Alarm feature 4: Euclidean distance consecutive change
        consecutive_distances_euclidean = np.array([
            np.linalg.norm(repr_extremity[i]-repr_extremity[i-1])
            for i in range(1,n_meps)
        ])

        # Alarm feature 5: Manhattan distance consecutive change
        consecutive_distances_manhattan = np.array([
            np.sum(np.abs(repr_extremity[i]-repr_extremity[i-1]))
            for i in range(1, n_meps)
        ])

        # Alarm feature 6: dot product consecutive change
        consecutive_dot = np.array([
            np.dot(repr_extremity[i],repr_extremity[i-1]) 
            for i in range(n_meps)
        ])

        # Alarm feature 7: Net displacement
        net_displacement = np.sum(np.abs(repr_extremity[-1]-baseline))

        # Combine all alarm features in a dataframe
        alarm_features_extremity.append({
            'patient_id':                       patient.astype(str),
            'side':                             side.astype(str),
            'outcome':                          outcome,
            'n_meps':                           n_meps,
            'auc_baseline_dist_euclidean':      float(normalized_auc(baseline_distances_euclidean)),
            'auc_baseline_dist_manhattan':      float(normalized_auc(baseline_distances_manhattan)),
            'auc_baseline_dot':                 float(normalized_auc(baseline_dot)),
            'auc_consecutive_dist_euclidean':   float(normalized_auc(consecutive_distances_euclidean)),
            'auc_consecutive_dist_manhattan':   float(normalized_auc(consecutive_distances_manhattan)),
            'auc_consecutive_dot':              float(normalized_auc(consecutive_dot)),
            'net_displacement':                 float(net_displacement),
        })

    alarm_features = pd.DataFrame(alarm_features_extremity)
    alarm_features = alarm_features.sort_values(['patient_id','side']).reset_index(drop=True)
    return alarm_features        


def compute_mep_alarm_timeseries(representations, patient_labels, side_labels, outcome_labels, epoch_ids):
    '''
    Compute alarm values per MEP throughout surgery.

    Input:
        representations  :  Learned 320-dimensional representations of each MEP
        patient_labels   :  Patient labels assigned to each MEP
        side_labels      :  Side labels assigned to each MEP
        outcome_labels   :  Outcome labels assigned to each MEP
        epoch_ids        :  Epoch id per MEP
    Output:
        alarm_timeseries : Dataframe containing the alarm features per MEP
    '''

    alarm_features_timeseries = []

    # Retrieve unique extremities by searching on the combination patient label and side label
    extremities = set(zip(patient_labels, side_labels)) 

    for patient, side in extremities:
        mask            =(patient_labels == patient) & (side_labels == side)
        sel_indices     = np.where(mask)[0]
        repr_extremity  = representations[sel_indices] 
        outcome         = int(outcome_labels[sel_indices][0]) 
        n_meps          = int(len(repr_extremity))

        if n_meps < 2:
            print(f'Skipping {patient} {side}: too few MEPs ({n_meps})')
            continue

        baseline = repr_extremity[0] # Take first MEP as baseline

        # Alarm feature 1: Euclidean distance relative to baseline
        baseline_distances_euclidean = np.array([
            np.linalg.norm(repr_extremity[i]-baseline)
            for i in range(n_meps)
        ])

        # Alarm feature 2: Manhattan distance relative to baseline
        baseline_distances_manhattan = np.array([
            np.sum(np.abs(repr_extremity[i]-baseline))
            for i in range(n_meps)
        ])

        # Alarm feature 3: dot product relative to baseline
        baseline_dot = np.array([
            np.dot(repr_extremity[i],baseline) # higher value = greate similarity
            for i in range(n_meps)
        ])

        # Alarm feature 4: Euclidean distance consecutive change
        consecutive_distances_euclidean = np.concatenate([
            [0], [np.linalg.norm(repr_extremity[i]-repr_extremity[i-1])
            for i in range(1,n_meps)]
        ])

        # Alarm feature 5: Manhattan distance consecutive change
        consecutive_distances_manhattan = np.concatenate([
            [0], [np.sum(np.abs(repr_extremity[i]-repr_extremity[i-1]))
            for i in range(1,n_meps)]
        ])

        # Alarm feature 6: dot product consecutive change
        consecutive_dot = np.array([
            np.dot(repr_extremity[i],repr_extremity[i-1]) 
            for i in range(n_meps)
        ])

        # Create dataframe containing alarm criteria per MEP
        for epoch_pos in range(n_meps):
            alarm_features_timeseries.append({
                'patient_id':                   patient.astype(str),
                'side':                         side.astype(str),
                'outcome':                      outcome,
                'epoch_id':                     epoch_ids[sel_indices[epoch_pos]].astype(str),
                'epoch_index':                  int(epoch_pos),
                'baseline_dist_euclidean':      float(baseline_distances_euclidean[epoch_pos]),
                'baseline_dist_manhattan':      float(baseline_distances_manhattan[epoch_pos]),
                'baseline_dot':                 float(baseline_dot[epoch_pos]),
                'consecutive_dist_euclidean':   float(consecutive_distances_euclidean[epoch_pos]),
                'consecutive_dist_manhattan':   float(consecutive_distances_manhattan[epoch_pos]),
                'consecutive_dot':              float(consecutive_dot[epoch_pos]),
            })

    alarm_timeseries = pd.DataFrame(alarm_features_timeseries)
    alarm_timeseries = alarm_timeseries.sort_values(['patient_id','side','epoch_index']).reset_index(drop=True)
    return alarm_timeseries  


def run_alarm_statistics(alarm_features, feats, side=None):
    '''
    Function to perform statistical analyses on alarm criteria.
    Perform Mann-Whitney U test. 

    Input:
        alarm_features (pd.DataFrame): Dataframe containing alarm criteria per extremity
        feats (list)                 : Alarm criteria you're interested in
        side (str, None)             : Side you're interested in
                                       If not specified, perform statistical analyses for all extremities
    Output:
        results_df (pd.DataFrame)    : Dataframe containing statistical results
    '''

    df = alarm_features.copy()

    # Select alarm criteria of the side(s) you're interested in
    if side is not None:
        df = df[df['side']==side].reset_index(drop=True)
    
    df[feats]        = df[feats].apply(pd.to_numeric, errors='coerce')
    df['patient_id'] = df['patient_id'].astype(str)
   
    def q1(x): return x.quantile(0.25)
    def q3(x): return x.quantile(0.75)
    
    # Print a summary of the alarm criteria (median, q1, q3)
    print(df.groupby('outcome')[['auc_baseline_dist_euclidean',
        'auc_baseline_dist_manhattan',
        'auc_baseline_dot',
        'auc_consecutive_dist_euclidean',
        'auc_consecutive_dist_manhattan',
        'auc_consecutive_dot',
        'net_displacement']].agg(['median',q1, q3]))
        
    results = []
    for feat in feats:

        group0 = df[df['outcome'] ==0][feat]
        group1 = df[df['outcome'] ==1][feat]

        # Determine whether the alarm criteria are normally distributed
        _, p0 = stats.shapiro(group0)
        _, p1 = stats.shapiro(group1)
        print(f'{feat}: Normal? group0: p={p0:.2f}, group1: p={p1:.2f}')

        stat_mw, pvalue_mw = stats.mannwhitneyu(group0,group1,alternative='two-sided')
        
        def rank_biserial(group0,group1,stat):
            # Perform rank biserial correlation to estimate effect sizes of Mann-Whitney U test
            # r is between -1 and 1. 0.1 = small effect, 0.3 = medium effect, 0.5 = large effect
            n0,n1 = len(group0), len(group1)
            r = 1-(2*stat)/(n0*n1)
            return r

        r = rank_biserial(group0,group1,stat_mw)
        print(f'{feat}: U={stat_mw:.2f}, p={pvalue_mw:.4f}, r={r:.3f}')

        results.append({
            'feature': feat,
            'mw_stat': round(stat_mw,4),
            'mw_p_value': round(pvalue_mw,4),
            'mw_effect_size_r': round(r,4),
        })

    results_df = pd.DataFrame(results)

    # Benjamini Hochberg correction for multiple testing
    reject_mw, p_adj_mw, _, _ = multipletests(results_df['mw_p_value'],alpha=0.05,method='bonferroni')
    results_df['mw_p_value_bonf'] = np.round(p_adj_mw, 4)
    results_df['mw_significant_bonf'] = reject_mw # True if test is significant after correction 
    print(results_df)
    return results_df
