'''
Module containing functions for visualization the supervised machine learning results
'''

###-------IMPORTING LIBRARIES------###
import matplotlib.pyplot as plt
import os
import numpy as np
import shap
import seaborn as sns
from sklearn.metrics import DetCurveDisplay, precision_recall_curve, average_precision_score
from sklearn.inspection import PartialDependenceDisplay
from PyALE import ale
from matplotlib.colors import LinearSegmentedColormap

# %% PLOTS FEATURES
def correlation(corr, resultfolder, file_png, file_svg, cluster_file_png, cluster_file_svg):
    '''
    Function to plot the correlation between features from different muscles
    '''
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
    heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=False, cmap="BrBG", ax=ax)
    plt.tight_layout()

    save_loc_png = os.path.join(resultfolder, file_png)
    save_loc_svg = os.path.join(resultfolder, file_svg)
    plt.savefig(save_loc_png)
    plt.savefig(save_loc_svg)
    plt.close()

    clustermap = sns.clustermap(
        corr, vmin=-1, vmax=1, annot=False, cmap="BrBG", figsize=(12, 12)
    )
    save_loc_cluster_png = os.path.join(resultfolder, cluster_file_png)
    save_loc_cluster_svg = os.path.join(resultfolder, cluster_file_svg)
    plt.savefig(save_loc_cluster_png)
    plt.savefig(save_loc_cluster_svg)
    plt.close()

# %% PLOTS CLASSIFIER 
def single_roc(resultfolder, mean_fpr, mean_tpr,tprs_lower, tprs_upper, mean_auc, std_auc):
    '''
    Function to plot a single ROC.
    Only titles of plots have been changed for this project (compared to the original file from M. Verboom)
    '''
    # Single ROC
    fig, ax = plt.subplots(figsize=[8, 8])
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    ax.plot(
        mean_fpr,
        mean_tpr,
        color='#5AA5DB',
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=3,
        alpha=0.8,
    )
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC curve \n(Positive label 'Postoperative motor decline', AUC = %0.2f)" % (mean_auc),
    )
    ax.axis("square")

    plt.savefig(resultfolder + "/ROC_Curve_mean.png")    
    plt.savefig(resultfolder + "/ROC_Curve_mean.svg")
    plt.close()

    # Single ROC with SD
    fig, ax = plt.subplots(figsize=[8, 8])
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    ax.plot(
        mean_fpr,
        mean_tpr,
        color='#5AA5DB',
        lw=3,
        alpha=0.8,
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    )
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
        title="Mean ROC curve",
    )
    ax.axis("square")
    ax.legend(loc="lower right")

    plt.savefig(resultfolder + "/ROC_Curve_mean_std.png") 
    plt.savefig(resultfolder + "/ROC_Curve_mean_std.svg")
    plt.close()

def visualize_cf_mat(cf_mat, resultfolder):
    '''
    Function to create an annotated heatmap of the confusion matrix
    Only titles of plots have been changed for this project (compared to the original file from M. Verboom)
    https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
    '''
    group_names         = [
                            "True negative",
                            "False positive",
                            "False negative",
                            "True positive",
                        ]
    group_counts        = ["{0:0.0f}".format(value) for value in cf_mat.flatten()]
    group_percentages   = [
                            "{0:.2%}".format(value) for value in cf_mat.flatten() / np.sum(cf_mat)]
    labels              = [
                            f"{v1}\n{v2}\n{v3}"
                            for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels              = np.asarray(labels).reshape(2, 2)

    # Calculate scores 
    accuracy            = np.trace(cf_mat) / float(np.sum(cf_mat))
    precision           = cf_mat[1, 1] / sum(cf_mat[:, 1])
    recall              = cf_mat[1, 1] / sum(cf_mat[1, :])
    f1_score            = 2 * precision * recall / (precision + recall)
    
    stats_text = (
        "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
            accuracy, precision, recall, f1_score
        )
    )

    plt.figure(figsize=(8, 8))

    custom_cmap         = LinearSegmentedColormap.from_list('custom_blue',['white','#5AA5DB'])
    sns.heatmap(cf_mat, annot=labels,annot_kws={"fontsize":14}, fmt="", cmap=custom_cmap, cbar=False)
    plt.ylabel("True label",fontsize=14)
    plt.xlabel("Predicted label", fontsize=14)
    plt.tight_layout()
    plt.savefig(resultfolder + "/Confusion_Matrix.png")
    plt.savefig(resultfolder + "/Confusion_Matrix.svg")
    plt.close()
 
# Feature importance
def shap_decision(expected_values, shap_values, x_test, feature_set,fold, resultfolder):
    '''
    SHAP decision plot adapted from M. Verboom
    Created per fold
    '''
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16, 12))
    shap.decision_plot(
        expected_values, 
        shap_values,
        x_test,
        feature_names=feature_set,
        show=False,
        alpha=0.75,
        ignore_warnings=True,
    )
    plt.tick_params(axis='y',labelsize=7)
    plt.tight_layout()
    plt.savefig(resultfolder + f"/SHAP_Decision_Plot_All_Epochs_fold{fold}.png") 
    plt.savefig(resultfolder + f"/SHAP_Decision_Plot_All_Epochs_fold{fold}.svg")
    plt.close()

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16, 12))
    shap.decision_plot(
        expected_values, 
        shap_values,
        x_test,
        feature_names=feature_set,
        show=False,
        alpha=0.75,
        feature_display_range=slice(-1, -11, -1),
        ignore_warnings=True,
    )
    plt.tick_params(axis='y',labelsize=10)
    plt.tight_layout()
    plt.savefig(resultfolder +f"/SHAP_Decision_Plot_All_Epochs_Top10_fold{fold}.png")
    plt.savefig(resultfolder + f"/SHAP_Decision_Plot_All_Epochs_Top10_fold{fold}.svg")
    plt.close()

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(16, 12))
    shap.summary_plot(
        shap_values,
        x_test,
        feature_names=feature_set,
        plot_type="bar",
        max_display=x_test.shape[1],
        show=False,
    )
    plt.tick_params(axis='y',labelsize=7)
    plt.tight_layout()
    plt.savefig(resultfolder + f"/SHAP_Decision_Summary_Plot_AllFeatures_fold{fold}.png")
    plt.savefig(resultfolder + f"/SHAP_Decision_Summary_Plot_AllFeatures_fold{fold}.svg")
    plt.close()

def shap_beeswarm(shap_values, x_test, feature_set, fold, resultfolder):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,6))
    explanation = shap.Explanation(
        values=shap_values,
        data=x_test.values,
        feature_names=feature_set
    )
    shap.plots.beeswarm(
        explanation, show=False, plot_size=(10,6)
    )
    plt.tick_params(axis='y',labelsize=10)
    plt.tight_layout(rect=[0,0,0.85,1])
    plt.savefig(resultfolder + f"/SHAP_Beeswarm_Summary_Plot_AllFeatures_fold{fold}.png")
    plt.savefig(resultfolder + f"/SHAP_Beeswarm_Summary_Plot_AllFeatures_fold{fold}.svg")
    plt.close()

def shap_scatter(shap_values, x_test, y_true, resultfolder, fold):
    explanation = shap.Explanation(
        values=shap_values,
        data=x_test.values,
        feature_names=x_test.columns.tolist()
    )

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top5_idx = np.argsort(mean_abs_shap)[-5:] 
    top5_features = x_test.columns[top5_idx].tolist()

    color_array = y_true.values.astype(float)
    for feature in x_test:
        fig, ax = plt.subplots(figsize=(8,5))
        shap.plots.scatter(
            explanation[:,feature],
            color=color_array,
            cmap=plt.get_cmap('coolwarm'),
            show=False,
            ax=ax
        )
        ax.tick_params(axis='x',labelsize=14)
        ax.tick_params(axis='y',labelsize=14)
        ax.set_xlabel('Feature value',fontsize=14)
        ax.set_ylabel('SHAP value',fontsize=14)
        ax.legend(fontsize=13)
        if fold is not None:
            ax.set_title(f'SHAP scatter: {feature} (fold {fold})',fontsize=14)

        if len(ax.figure.axes)>1:
            ax.figure.axes[-1].remove()
        plt.tight_layout()
        if fold is not None:
            plt.savefig(resultfolder + f"/SHAP_Scat_{feature}_f{fold}.png")
        else:
            plt.savefig(resultfolder+f'/SHAP_Scatter_Feature_{feature}_all_folds.png')
        plt.close()

def ice_plot(classifier, x_test, selected_features, fold, resultfolder):
    for feature in selected_features:
        fig, ax = plt.subplots(figsize=(8,5))
        PartialDependenceDisplay.from_estimator(
            classifier,
            x_test,
            features=[feature],
            kind='both', 
            ax=ax,
        )
        for line in ax.lines:
            line.set_color('#5AA5DB')
        ax.set_xlabel(ax.get_xlabel(),fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        ax.tick_params(axis='x',labelsize=14)
        ax.tick_params(axis='y',labelsize=14)
        plt.tight_layout()
        plt.savefig(resultfolder + f'/ICE_{feature}_fold{fold}.png')
        plt.close()

class ProbaWrapper:
    def __init__(self,model):
        self.model=model
    def predict(self, X):
        return self.model.predict_proba(X)[:,1]

def ale_plot(features, ale_data, resultfolder, fold=None):
    fold_str = f'_fold{fold}' if fold is not None else ''
    for feat in features:
        curves =[]
        for fold in ale_data:
            wrapped_model = ProbaWrapper(fold['model'])
            try:
                result = ale(
                    X=fold['X'],
                    model=wrapped_model,
                    feature=[feat],
                    feature_type='continuous',
                    plot=False,
                    grid_size=20
                )

                curves.append(result)
            except Exception as e:
                print(f'ALE failed for {feat} in fold: {e}')
    
        # mean curve across folds
        fig,ax = plt.subplots(figsize=(6,4))

        for c in curves:
            ax.plot(c.index, c['eff'],alpha=0.3,color='#5AA5DB')
        if curves: 
            common_x = np.linspace(
                min(c.index.min() for c in curves),
                max(c.index.max() for c in curves),
                50
            )
            mean_eff = np.mean(
                [np.interp(common_x, c.index, c['eff']) for c in curves],
                axis=0
            )
            ax.plot(common_x, mean_eff, color='#5AA5DB',linewidth=2,label=f'Average ({len(curves)} folds)')
        ax.axhline(0,color='black',linewidth=0.8,linestyle='--')
        ax.tick_params(axis='x',labelsize=14)
        ax.tick_params(axis='y',labelsize=14)
        ax.set_ylabel('ALE', fontsize=14)
        ax.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(resultfolder + f'/ALE_{feat}{fold_str}.png')
        plt.close()
