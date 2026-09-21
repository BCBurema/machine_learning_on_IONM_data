'''
Module to fit the supervised machine learning model using stratified group k-fold cross-validation.

Adapted from Hunfeld et al. (2024) (doi: 10.1212/WNL.0000000000210043)
'''

# %% Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import shap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from TM3_Internship_BB.supervised_ml.rf_classifier_functions import metrics_classifier, plot_cv_indices, adjusted_classes
from TM3_Internship_BB.supervised_ml.visualization_sml import shap_decision, shap_beeswarm, shap_scatter, ice_plot, ale_plot
from TM3_Internship_BB.supervised_ml.datautils_sml import train_feature_selection
from collections import defaultdict


# %% 
def stratified_group_k_fold(n_splits, PatID, X, y, resultfolder):
    '''
    Perform stratified group k-fold cross-validation and visualize splitting.

    Input:
        n_splits (int)          : Number of splits you'd like to make
        PatId (pd.DataFrame)    : Dataframe containing all participant ids
        X (pd.DataFrame)        : Dataframe with preselected features
        y (pd.DataFrame)        : Outcome labels
        resultfolder (path)     : Path to resultfolder
    Output:
        sgkf (object)           : Cross-validation splitter
    '''

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=26)

    # Create plot
    cmap_data       = plt.cm.Paired
    cmap_cv         = plt.cm.coolwarm
    vals            = np.linspace(0, 1, 256)
    np.random.shuffle(vals)
    cmap_group      = plt.cm.colors.ListedColormap(plt.cm.jet(vals))
    cmap_outcome    = LinearSegmentedColormap.from_list("gr", ["g", "w", "r"], N=5)

    fig, ax         = plt.subplots(figsize=(12, 6))
    plot_cv_indices(sgkf, X, y, PatID, ax, n_splits, cmap_group, cmap_outcome, lw=10)
    ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ["Test set", "Training set"],
        loc=(1.02, 0.8),
    )

    # Make the legend fit
    plt.tight_layout()
    fig.subplots_adjust(right=0.7)
    plt.savefig(resultfolder + "/KFold_Split.png")
    plt.savefig(resultfolder + "/KFold_Split.svg")
    plt.close()
    return sgkf

def fit_model(X, muscles, y, PatID, classifier, n_splits, param_grids, anova_k, threshold, not_per_muscle, scorers, refit_score, resultfolder, plot=False): 
    '''
    Function to fit the specified classifier.

    Input:
        X (pd.DataFrame)        : Dataframe with preselected features
        muscles (list)          : List of muscles analyzed
        y (pd.DataFrame)        : Outcome labels
        PatId (pd.DataFrame)    : Dataframe containing all participant ids
        classifier (str)        : Classifier you're interested in
        n_splits (int)          : Number of splits you'd like to make
        param_grids (dict)      : Dictionary containing dictionaries per classifier -> containing parameter grids
        anova_k (int)           : Number of features that should be selected in each fold
        threshold (int)         : Threshold to assign positive or negative label
        not_per_muscle (bool)   : If True, aggregate features from muscles to extremity level. Else: use features from muscles as input for classifier
        scorers (dict)          : Dictionary containing evaluation metrics
        refit_score (str)       : Score that should be optimized to find the best hyperparameters
        resultfolder (path)     : Path to resultfolder
        plot (bool)             : Whether or not to create plots
    Output:
        fold_scores (list)          : Evaluation metrics per fold
        x_test_parts (list)         : Per fold a dataframe with the test data, restricted to the selected features
        x_test_full (pd.DataFrame)  : Dataframe combining x_test_parts; NaN where a feature was not selected in a fold
        y_true (list)               : Per fold, a list with the true test labels (0=stability/improvement, 1=deterioration)
        y_chance (list)             : Per fold, a list with the predicted probability of the positive class
        y_predict (list)            : Per fold, the predicted labels (0 or 1) after applying the threshold
        mean_fpr (np.ndarray)       : Fixed FPR axis
        mean_tpr (np.ndarray)       : Mean interpolated TPR across folds
        tprs_lower (np.ndarray)     : mean_tpr - 1 standard deviation
        tprs_upper (np.ndarray)     : mean_tpr + 1 standard deviation
        mean_auc (float)            : AUC of the mean ROC curve
        std_auc (float)             : Standard deviation of the per-fold AUCs
        list_shap_values (list)     : Per fold, an np.ndarray with SHAP values
        list_test_sets (list)       : Per fold, the test index from the sgkf
        list_expected_values (list) : Per fold, the TreeExplainer expected value 
        list_features (list)        : Per fold, a list of str with the selected feature names
        stable_features (list)      : List of str with features selected in all folds
        fold_data (list)            : Per fold, a dict with keys 'model' (best estimator), 'X' (x_test_sel), 'features' (list of str), 'shap_values' (np.ndarray), 'fold' (int)
    '''

    sgkf                     = stratified_group_k_fold(n_splits, PatID, X, y, resultfolder)
    y_true                  = []
    y_predict               = []
    y_chance                = []
    x_test_parts            = []
    PID_test                = []
    rf_accu_stratified      = [] # misschien iets anders kiezen??
    list_shap_values        = list()
    list_test_sets          = list ()
    list_expected_values    = list()
    list_features           = list()
    fold_data               = [] # list of dictionaries, added per fold
    tprs                    = []
    aucs                    = []
    mean_fpr                = np.linspace(0,1,100)
    fold_scores             = []

    # Initialize figure for ROC plot
    fig, ax                 = plt.subplots(figsize=[8,8])
    ax.plot([0,1],[0,1],'k--',label='chance level (AUC=0.5)') # wat is dit??

    # All possible models
    models = {
        'RF':   RandomForestClassifier(class_weight='balanced', random_state=26),
        'BRF':  BalancedRandomForestClassifier(sampling_strategy='all',replacement=True,bootstrap=False,class_weight=None,random_state=26),
        'XGB':  XGBClassifier(scale_pos_weight=14, random_state=26, eval_metric='logloss')
    }

    results     = []
    for fold, (train_index, test_index) in enumerate(sgkf.split(X, y, PatID)):
        # Dependent on which classifier has been selected
        if classifier == 'xgb':
            classifier_optimized = RandomizedSearchCV( 
                models['XGB'], 
                param_grids['XGB'],
                n_iter=30,
                cv=5,
                verbose=0, 
                random_state=26,
                n_jobs=1,
                scoring=scorers,
                refit=refit_score,
                error_score='raise',
                return_train_score=True
            )
        elif classifier == 'rf':
            classifier_optimized = RandomizedSearchCV( 
                models['RF'], 
                param_grids['RF'],
                n_iter=30,
                cv=5,
                verbose=0, 
                random_state=26,
                n_jobs=1,
                scoring=scorers,
                refit=refit_score,
                error_score='raise',
                return_train_score=True
            )
        elif classifier == 'brf':
            classifier_optimized = RandomizedSearchCV( 
                models['BRF'], 
                param_grids['BRF'],
                n_iter=30,
                cv=5,
                verbose=0, 
                random_state=26,
                n_jobs=1,
                scoring=scorers,
                refit=refit_score,
                error_score='raise',
                return_train_score=True
            )


        ### SPLIT DATA IN TRAIN AND TEST SET
        x_train_fold, x_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
        
        print(f'Fold {fold}: postoperative motor deterioration train set = {sum(y_train_fold)}, postoperative motor deterioration test set = {sum(y_test_fold)}')
        print(f'Fold {fold}: postoperative stability/improvement train set = {len(y_train_fold) - sum(y_train_fold)}, postoperative stability/improvement test set = {len(y_test_fold) - sum(y_test_fold)}')
        
        ### FEATURE SELECTION
        x_train_sel, selected_features = train_feature_selection(muscles,x_train_fold, y_train_fold, anova_k=anova_k, corr_threshold=0.8, not_per_muscle=not_per_muscle)
        
        ### FIT MODEL
        classifier_optimized.fit(x_train_sel, y_train_fold)
        score       = classifier_optimized.best_score_
        
        x_test_sel  = x_test_fold[selected_features] 
        x_test_parts.append(x_test_sel)
        list_features.append(selected_features)

        print("Best params for {}".format(refit_score))
        print(classifier_optimized.best_params_) 
        results = pd.DataFrame(classifier_optimized.cv_results_) 
        results = results.sort_values(by="mean_test_recall_score", ascending=False) 
        if classifier == 'xgb':
            results[            
                [
                    "mean_test_f2_score",
                    "mean_test_precision_score",
                    "mean_test_recall_score",
                    "mean_test_accuracy_score",
                    "param_max_depth",
                    "param_min_child_weight",
                    "param_n_estimators",
                ]
            ].round(3).head() 
        else: # RF en BRF
            results[            
                [
                    "mean_test_f2_score",
                    "mean_test_precision_score",
                    "mean_test_recall_score",
                    "mean_test_accuracy_score",
                    "param_max_depth",
                    "param_min_samples_split",
                    "param_n_estimators",
                ]
            ].round(3).head() 

        ### PRINT RESULTS
        best_idx = classifier_optimized.best_index_ 
        train_score = classifier_optimized.cv_results_['mean_train_recall_score'][best_idx]
        test_score = classifier_optimized.cv_results_['mean_test_recall_score'][best_idx] 
        print(f'Train recall: {train_score.mean():.3f}, test recall: {test_score.mean():.3f}')

        best_classifier = classifier_optimized.best_estimator_
        y_scores        = best_classifier.predict_proba(x_test_sel)[:, 1]   # Predict proba provides percentage for each possible class -> using a threshold, you can assign label 0 or 1

        y_predict_fold  = adjusted_classes(y_scores, threshold)           

        ### CONFUSION MATRIX ON TEST SET
        print(
            "\nConfusion matrix of Random Forest optimized for {} on the test data:".format(
                refit_score
            )
        )
        print(
            pd.DataFrame(
                metrics.confusion_matrix(y_test_fold, y_predict_fold, labels=[0,1]), 
                columns=["pred_neg", "pred_pos"],
                index=["neg", "pos"],
            )
        )

        rf_accu_stratified.append(classifier_optimized.score(x_test_sel, y_test_fold)) 
        y_true.append(y_test_fold)
        y_chance.append(y_scores)
        y_predict.append(y_predict_fold)
        PID_test.append(PatID.iloc[test_index])

        # Save the test values to use in combination with Shapley and y_true / y_predict
        fold_scores.append(metrics_classifier(y_test_fold, y_predict_fold,y_scores))

        # For each fold, calculate Shapley values and combine
        explainer = shap.TreeExplainer(best_classifier, data=x_train_sel)
        if isinstance(explainer.expected_value, np.ndarray) and len(explainer.expected_value) >1:
            expected_value = explainer.expected_value[1]
            print(f'expected value: {expected_value}')
        else:
            expected_value = explainer.expected_value
            print(f'expected value: {expected_value}')
       
        shap_values = explainer.shap_values(x_test_sel)

        # Shap plots per fold
        if classifier == 'xgb':
            sv = shap_values
        elif isinstance(shap_values, list):
            sv = shap_values[1]
        else: 
            sv = shap_values[:,:,1]

        if plot:
            shap_decision(expected_value, sv, x_test_sel, selected_features, fold, resultfolder)
            shap_beeswarm(sv, x_test_sel, selected_features, fold, resultfolder)
            shap_scatter(sv, x_test_sel, y_test_fold, resultfolder, fold)

            top5_shap = x_test_sel.columns[np.argsort(np.abs(sv).mean(axis=0))[-5:]].tolist()
            ice_plot(best_classifier, x_test_sel, selected_features, fold, resultfolder)

            ale_plot(selected_features, [{ 
                'model':     best_classifier,
                'X':         x_test_sel,
                'features':  selected_features}], resultfolder, fold=fold)

        # Collect data for ALE plot 
        fold_data.append({
            'model':        classifier_optimized.best_estimator_, 
            'X':            x_test_sel,
            'features':     selected_features, 
            'shap_values':  sv,
            'fold':         fold,
            })

        # for each iteration we save the test_set index and the shap_values
        list_shap_values.append(sv)
        list_expected_values.append(expected_value)
        list_test_sets.append(test_index)

        # Create ROC curve per fold
        fold_colors = ["#8734C8","#CF8126","#60BF57","#FCFF67","#F45B5B"]
        viz = metrics.RocCurveDisplay.from_estimator(
            classifier_optimized, 
            x_test_sel,
            y_test_fold,
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
            color=fold_colors[fold]
        )

        interp_tpr      = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0]   = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    
    x_test_full = pd.concat(x_test_parts, ignore_index=True)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color='#5AA5DB',
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color='#5AA5DB',
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC curve with variability\n(Positive label 'Postoperative motor decline')",
    )
    ax.xaxis.label.set_size(14)
    ax.xaxis.label.set_size(14)
    ax.title.set_size(14)
    ax.tick_params(axis='x',labelsize=14)
    ax.tick_params(axis='y',labelsize=14)
    ax.axis("square")
    ax.legend(loc="lower right",fontsize=13)
    plt.savefig(resultfolder + "/ROC_Curve_folds.png")
    plt.savefig(resultfolder + "/ROC_Curve_folds.svg")
    plt.close()

    # Determine stable features
    feature_fold_counts = defaultdict(int)
    for fold in fold_data:
        for f in fold['features']: 
            feature_fold_counts[f] += 1
    stable_features = [f for f, count in feature_fold_counts.items() if count>=len(fold_data)-1] # check which features are present in every fold
    print(f'{len(stable_features)} features present in all 5 folds.')
    print(stable_features)
    return fold_scores, x_test_parts, x_test_full, y_true, y_chance, y_predict, mean_fpr, mean_tpr, tprs_lower, tprs_upper, mean_auc, std_auc, list_shap_values, list_test_sets, list_expected_values, list_features, stable_features, fold_data
