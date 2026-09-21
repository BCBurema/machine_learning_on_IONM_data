'''
This module contains a function to visualize the distribution of number of epochs per participant.

Author: B.C. Burema
Date: 17/07/2026
'''

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import pandas as pd

# %% Plotten   
def plot_number_epochs(records, remove_participants=None):
    '''
    Create an histogram visualizing the distribution of number of epochs per participant.
    
    Input:
        records (list)                      :    List containing dictionaries per participant with original data.
                                                Each dictionary contains participant_id, epochs (EpochsFIF), number of epochs.
        remove_participants (None or list)  :   None, or list containing participant_ids that should be removed from the image.
    '''
    # Create a dataframe of the records
    records_df = pd.DataFrame(records)

    if remove_participants is not None:
        records_df = records_df[~records_df['participant_id'].isin(remove_participants)]    
    
    # Initialize the x-axis with number of epochs
    max_counts = records_df['n_epochs'].max()
    bins = np.arange(0, max_counts+11, 10)
    epoch_counts, edges = np.histogram(records_df['n_epochs'], bins=bins)
    labels = [f'{int(edges[i])}-{int(edges[i+1])}' for i in range(len(edges)-1)]

    # Create figure
    fig, ax = plt.subplots(figsize=(10,5))

    ax.bar(labels, epoch_counts, color='#5AA5DB', width=0.7)
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('Number of participants')
    ax.set_title('Distribution of number of MEP epochs per participant')

    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.xticks(rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'TM3_Internship_BB/distribution_epochs.png')
    plt.show()
