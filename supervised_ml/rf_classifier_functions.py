'''
RF classifier functions for analysis and visualization
Companion file to keep the main file (rf_classifier) better readable.

Adapted from Hunfeld et al. (2024) (doi: 10.1212/WNL.0000000000210043)
'''

###------IMPORTING LIBRARIES------###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

###------DEFINE CUSTOM FUNCTIONS------###
# Metrics classifier, handy overview of performance metrics (work by Femke Lückerath)
# Adapted to metrics I was interested in for my thesis
def metrics_classifier(test_label, pred_label, chance_label):
    scores = pd.DataFrame(
        columns=[
            "Sensitivity",
            "F2-score",
            "Specificity",
            "ROC-AUC",
        ],
        index=["Value"],
    )
    cm = metrics.confusion_matrix(test_label, pred_label)

    scores["Sensitivity"]   = metrics.recall_score(test_label, pred_label)
    scores['F2-score']      = metrics.fbeta_score(test_label, pred_label, beta=2, zero_division=0)
    scores["Specificity"]   = specificity_score(test_label, pred_label)
    scores["ROC-AUC"]       = metrics.roc_auc_score(test_label, chance_label)
    return cm, scores

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp  = metrics.confusion_matrix(y_true, y_pred).ravel()
    spec            = tn / (tn + fp)
    return spec

def adjusted_classes(y_scores, t):
    '''
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.
    '''
    return [1 if y >= t else 0 for y in y_scores]

def plot_cv_indices(cv, X, y, group, ax, n_splits, cmap_group, cmap_outcome, lw=10):
    '''
    Create a sample plot for indices of a cross-validation object.
    '''

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices     = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data classes and groups at the end
    ax.scatter(
        range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_outcome
    )

    ax.scatter(
        range(len(X)), [ii + 2.5] * len(X), c=pd.factorize(group)[0], marker="_", lw=lw, cmap=cmap_group
    )

    # Formatting
    yticklabels = list(range(n_splits)) + ["class", "group"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
        ylim=[n_splits + 2.2, -0.2]
    )
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    return ax
