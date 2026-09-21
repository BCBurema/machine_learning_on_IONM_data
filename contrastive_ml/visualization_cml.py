'''
Module containing function to create t-SNE or UMAP plots.

Author: B.C. Burema
Date: 21/09/2026
'''

###------IMPORTING LIBRARIES------###
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.cm import plasma
import numpy as np
import pandas as pd
import os 
import umap.umap_ as umap
import matplotlib.colors as mcolors
import colorcet as cc
import psutil
import umap.plot

###------FUNCTION TO PLOT TSNE OR UMAP------###           
def tsne_plot(
    path, 
    repr,
    labels, 
    dimension_tsne, 
    random_seed, 
    all_patients_visualized=False, 
    outcome_labels=None, 
    side_labels=None, 
    center_labels=None, 
    surgical_durations=None,
    specified_patient=None,
    feature_labels=None, 
    feature_name=None,
    use_umap=False): 
    '''
    Plot a t-SNE visualization of the learned representations.

    Input:  AANPASSEN!!
        path (str)                                  : Path to save the figures to.
        repr (np.ndarray)                           : Encoded representations of all epochs of all patients
        labels (np.ndarray)                         : Containing patient ids repeated according to the number of epochs per patient
        dimension_tsne (int)                        : Either 2 or 3 for the t-SNE dimensionality reduction
        random_seed (int)                           : Random seed for reproducibility
        all_patient_visualized (bool)               : True or False; whether to assign each patient an unique color or not (default: False)     
        ...labels (array or None)                   : Arrays containing labels to color the embeddings
        use_umap (bool)                             : True or False; whether to use UMAP or t-SNE      
    '''

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if labels.ndim == 2:
        repr    = merge_dim01(repr)
        labels  = merge_dim01(labels)
    
    method_name = 'umap' if use_umap else 't-SNE'

    # Creating t-SNE or UMAP representations
    if use_umap:
        try:
            repr_umap=repr.astype(np.float32)

            reducer = umap.UMAP(
                n_components=dimension_tsne,
                n_neighbors=30,
                min_dist=0.3,
                metric='euclidean',
                random_state=random_seed,
                n_jobs=1,
                verbose=True,
                init='random',
                n_epochs=500
            )

            mapper = reducer.fit(repr_umap)
        except Exception as e:
            print(f'UMAP error: {e}')
            import traceback
            traceback.print_exc()
            raise
    else:
        if dimension_tsne   == 3:
            tsne            = TSNE(n_components=3, random_state=random_seed, perplexity=30)
            results         = tsne.fit_transform(repr)
        elif dimension_tsne == 2:
            tsne            = TSNE(n_components=2, random_state=random_seed, perplexity=30)
            results         = tsne.fit_transform(repr)
        np.save(f'{path}/tsne_results.npy',results)

    # Assign epochs a color depending on the type of coloring chosen
    if all_patients_visualized:
        if use_umap:
            unique_patients = np.unique(labels)
            n_patients      = len(unique_patients)
            palette         = cc.glasbey[:n_patients]
            color_key       = {pid: palette[i] for i,pid in enumerate(unique_patients)}
            ax              = umap.plot.points(mapper,labels=labels,color_key=color_key,background='white')#,width=1000,height=800)
            ax.set_title(f'{method_name} visualization')

            legend          = ax.get_legend()
            if legend is not None:
                legend.remove()
            ax.figure.savefig(f'{path}/{method_name}_plot_all_patients.png',dpi=150)
            plt.close()
        else:
            unique_patients = np.unique(labels)
            n_patients      = len(unique_patients)

            patient_to_int  = {p: i for i, p in enumerate(unique_patients)}
            color_indices   = np.array([patient_to_int[p] for p in labels])

            palette         = cc.glasbey[:n_patients]
            cmap_custom     = mcolors.ListedColormap(palette)

            if dimension_tsne == 3:
                fig           = plt.figure(figsize=(10,8))
                ax            = fig.add_subplot(111, projection='3d')
                scatter       = ax.scatter(results[:,0], results[:,1], results[:,2], c=color_indices, cmap=cmap_custom, s=5, alpha=0.7) # c was eerst labels

                ax.set_title(f'{method_name} visualization')
                fig.savefig(f'{path}/{method_name}_plot_all_patients.png')
                plt.close()
    
            if dimension_tsne == 2:
                fig           = plt.figure(figsize=(10,8))
                scatter       = plt.scatter(results[:,0], results[:,1], c=color_indices, cmap=cmap_custom, s=5, alpha=0.7) # c was eerst labels.astype(float).tolist(),

                plt.title(f'{method_name} visualization')
                fig.savefig(f'{path}/{method_name}_plot_all_patients.png')
                plt.close()

    if side_labels is not None:
        if use_umap:
            color_key   = {'left':'#5AA5DB','right':'#CF8126'}
            ax          = umap.plot.points(
                mapper,
                labels=side_labels,
                color_key=color_key,
                background='white'
            )
            ax.set_title(f'{method_name} visualization')
            ax.figure.savefig(f'{path}/{method_name}_plot_sides.png')
        else:
            colors = np.array(['#5AA5DB' if x=='left' else '#CF8126' for x in side_labels])
            legend = [
                Line2D([0],[0],marker='o', color='w', label='Right side', markerfacecolor='#CF8126' , markersize=10),
                Line2D([0],[0],marker='o', color='w', label='Left side', markerfacecolor='#5AA5DB', markersize=10),
            ]
            if dimension_tsne == 3:
                fig           = plt.figure(figsize=(10,8))
                ax            = fig.add_subplot(111, projection='3d')
                scatter       = ax.scatter(results[:,0], results[:,1], results[:,2], c=colors, s=5, alpha=0.7)

                ax.set_title(f'{method_name} visualization')
                ax.legend(handles=legend, loc='best')
                fig.savefig(f'{path}/{method_name}_plot_sides.png')
                plt.close()
    
            if dimension_tsne == 2:
                fig           = plt.figure(figsize=(10,8))
                scatter       = plt.scatter(results[:,0], results[:,1], c=colors, s=5, alpha=0.7)

                plt.title(f'{method_name} visualization')
                plt.legend(handles=legend, loc='best')
                fig.savefig(f'{path}/{method_name}_plot_sides.png')
                plt.close()

    if center_labels is not None:
        if use_umap:
            color_key   = {'GOSH':'#5AA5DB','EMC':'#CF8126'}
            ax          = umap.plot.points(
                mapper,
                labels=center_labels,
                color_key=color_key,
                background='white'
            )
            ax.set_title(f'{method_name} visualization')
            ax.figure.savefig(f'{path}/{method_name}_plot_center.png')
        else:
            colors = np.array(['#5AA5DB' if x=='GOSH' else '#CF8126' for x in center_labels])
            legend = [
                Line2D([0],[0],marker='o', color='w', label='EMC', markerfacecolor='#CF8126' , markersize=10),
                Line2D([0],[0],marker='o', color='w', label='GOSH', markerfacecolor='#5AA5DB', markersize=10),
            ]
            if dimension_tsne == 3:
                fig           = plt.figure(figsize=(10,8))
                ax            = fig.add_subplot(111, projection='3d')
                scatter       = ax.scatter(results[:,0], results[:,1], results[:,2], c=colors, s=5, alpha=0.7)

                ax.set_title(f'{method_name} visualization')
                ax.legend(handles=legend, loc='best')
                fig.savefig(f'{path}/{method_name}_plot_center.png')
                plt.close()
    
            if dimension_tsne == 2:
                fig           = plt.figure(figsize=(10,8))
                scatter       = plt.scatter(results[:,0], results[:,1], c=colors, s=5, alpha=0.7)

                plt.title(f'{method_name} visualization')
                plt.legend(handles=legend, loc='best')
                fig.savefig(f'{path}/{method_name}_plot_center.png')
                plt.close()
                
    if outcome_labels is not None:
        if use_umap:
            label_map               = {0: 'No change in MRC score', 1: 'Decrease in MRC score'}
            outcome_labels_names    = np.array([label_map[x] for x in outcome_labels])
            color_key               = {'No change in MRC score': '#60BF57','Decrease in MRC score':'#CF2626'}
            ax                      = umap.plot.points(
                mapper,
                labels=outcome_labels_names,
                color_key=color_key,
                background='white'
            )
            ax.set_title(f'{method_name} visualization')
            ax.figure.savefig(f'{path}/{method_name}_plot_outcomes.png')
        else:
            colors = np.array(['gray' if pd.isna(x) else ('#60BF57' if x==0 else '#CF2626') for x in outcome_labels])
            legend = [
                Line2D([0],[0],marker='o', color='w', label='No change in MRC score', markerfacecolor='#60BF57', markersize=10),
                Line2D([0],[0],marker='o', color='w', label='Decrease in MRC score', markerfacecolor='#CF2626', markersize=10),
            ]
            if dimension_tsne == 3:
                fig           = plt.figure(figsize=(10,8))
                ax            = fig.add_subplot(111, projection='3d')
                scatter       = ax.scatter(results[:,0], results[:,1], results[:,2], c=colors, s=5, alpha=0.7)

                ax.set_title(f'{method_name} visualization')
                ax.legend(handles=legend, loc='best')
                fig.savefig(f'{path}/{method_name}_plot_outcomes.png')
                plt.close()
    
            if dimension_tsne == 2:
                fig           = plt.figure(figsize=(10,8))
                scatter       = plt.scatter(results[:,0], results[:,1], c=colors, s=5, alpha=0.7)

                plt.title(f'{method_name} visualization')
                plt.legend(handles=legend, loc='best')
                fig.savefig(f'{path}/{method_name}_plot_outcomes.png')
                plt.close()

    if surgical_durations is not None:
        bins                = [0, 120, 180, 240, 300, 360, 1000]
        bin_labels          = ['0-2h','2-3h','3-4h','4-5h','5-6h','>6h']
        surgical_durations = np.array([td.total_seconds()/60 for td in surgical_durations])

        if use_umap:
            bin_indices     = np.digitize(surgical_durations,bins)-1
            bin_indices     = np.clip(bin_indices,0,len(bin_labels)-1)
            duration_labels = np.array([bin_labels[i] for i in bin_indices])

            duration_palette    = plt.get_cmap('viridis',len(bin_labels))(np.linspace(0,1,len(bin_labels)))
            color_key           = {label: duration_palette[i] for i,label in enumerate(bin_labels)}

            #plot1
            ax=umap.plot.points(
                mapper,
                labels=duration_labels,
                color_key=color_key,
                background='white'
            )
            ax.set_title(f'{method_name} visualization')
            ax.figure.savefig(f'{path}/{method_name}_plot_duration.png',dpi=150)
            plt.close(ax.figure)

            #plot2
            for outcome_val, outcome_name in [(0,'stable'),(1,'deterioration')]:
                mask = outcome_labels ==outcome_val
                ax=umap.plot.points(
                    mapper,
                    labels=duration_labels,
                    color_key=color_key,
                    subset_points=mask,
                    background='white'
                )
                ax.set_title(f'Outcome {outcome_val} ({outcome_name}), colored by surgical duration')
                ax.figure.savefig(f'{path}/{method_name}_plot_duration_outcome{outcome_val}.png',dpi=150)
                plt.close(ax.figure)
        else:
            cmap = plt.get_cmap('viridis',len(bin_labels))
            norm = mcolors.BoundaryNorm(bins,cmap.N)
            
            if dimension_tsne == 3:
                fig           = plt.figure(figsize=(10,8))
                ax            = fig.add_subplot(111, projection='3d')
                scatter       = ax.scatter(results[:,0], results[:,1], results[:,2], c=surgical_durations, cmap=cmap, norm=norm, s=5, alpha=0.7)

                ax.set_title(f'{method_name} visualization')
                cbar = fig.colorbar(scatter,ax=ax,shrink=0.6,ticks=[(bins[i]+bins[i+1])/2 for i in range(len(bin_labels)-1)]+[bins[-2]+30])
                cbar.ax.set_yticklabels(bin_labels)
                cbar.set_label('Surgical duration')
                fig.savefig(f'{path}/{method_name}_plot_duration.png')
                plt.close()
    
            if dimension_tsne == 2:
                fig           = plt.figure(figsize=(10,8))
                scatter       = plt.scatter(results[:,0], results[:,1], c=surgical_durations, cmap=cmap, norm=norm, s=5, alpha=0.7)

                plt.title(f'{method_name} visualization')
                cbar = plt.colorbar(scatter,shrink=0.6,ticks=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-2)]+[bins[-2]+30])
                cbar.ax.set_yticklabels(bin_labels)
                cbar.set_label('Surgical duration')
                fig.savefig(f'{path}/{method_name}_plot_duration.png')
                plt.close()

                mask_0      = outcome_labels ==0
                mask_1      = outcome_labels ==1

                fig2,axes   = plt.subplots(1,2,figsize=(16,7))

                axes[0].scatter(results[mask_1,0],results[mask_1,1], c='lightgray',s=5,alpha=0.7,zorder=0)

                scatter0    = axes[0].scatter(results[mask_0,0],results[mask_0,1],c=surgical_durations[mask_0],cmap=cmap,norm=norm,s=10,zorder=1)
                axes[0].set_title('Outcome 0 (stable), colored by surgical duration')
                cbar0       = fig2.colorbar(scatter0,ax=axes[0],shrink=0.6, ticks=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-2)]+[bins[-2]+30])
                cbar0.ax.set_yticklabels(bin_labels)

                axes[1].scatter(results[mask_0,0],results[mask_0,1],c='lightgray',s=5,alpha=0.7,zorder=0)
                scatter1    = axes[1].scatter(results[mask_1,0],results[mask_1,1],c=surgical_durations[mask_1],cmap=cmap,norm=norm,s=10,zorder=1)
                cbar1       = fig2.colorbar(scatter1,ax=axes[1],shrink=0.6,ticks=[(bins[i]+bins[i+1])/2 for i in range(len(bins)-2)]+[bins[-2]+30])
                cbar1.ax.set_yticklabels(bin_labels)
                plt.tight_layout()
                plt.savefig(f'{path}/tsne_surgery_duration_by_outcome.png')
                plt.close()

    if feature_labels is not None:
        # Clip feature labels to ignore outliers
        vmin            = np.nanpercentile(feature_labels, 5)
        vmax            = np.nanpercentile(feature_labels, 95)
        feature_labels = np.clip(feature_labels, vmin,vmax)
        if use_umap:
            print(f'{feature_name}: {feature_labels}')
            ax = umap.plot.points(
                mapper,
                values=feature_labels,
                cmap='viridis',
                background='white'
            )

            ax.set_title(f'{method_name} visualization')
            sm      = plt.cm.ScalarMappable(cmap='viridis',norm=plt.Normalize(vmin=vmin,vmax=vmax))
            cbar    = ax.figure.colorbar(sm,ax=ax,shrink=0.6)
            cbar.set_label(feature_name)
            ax.figure.savefig(f'{path}/{method_name}_plot_{feature_name}',dpi=150)
            plt.close(ax.figure)
        else:
            if dimension_tsne ==3:
                fig     = plt.figure(figsize=(10,8))
                ax      = fig.add_subplot(111,projection='3d')
                scatter = ax.scatter(results[:,0],results[:,1],results[:,2],c=feature_labels,cmap='viridis',s=5,alpha=0.7)
                ax.set_title(f'{method_name} visualization')
                cbar    = fig.colorbar(scatter,ax=ax,shrink=0.6,extend='both')
                cbar.set_label(feature_name)
                fig.savefig(f'{method_name}_plot_{feature_name}.png')
                plt.close()
            if dimension_tsne==2:
                fig     = plt.figure(figsize=(10,8))
                scatter = plt.scatter(results[:,0],results[:,1],c=feature_labels,cmap='viridis',s=5,alpha=0.7)
                plt.title(f'{method_name} visualization')
                cbar    = plt.colorbar(scatter,shrink=0.6,extend='both')
                cbar.set_label(feature_name)
                plt.savefig(f'{method_name}_plot_{feature_name}.png')
                plt.close()

    if specified_patient is not None:
        gray_mask   = labels != specified_patient
        color_mask  = labels == specified_patient

        if dimension_tsne == 3:
            fig           = plt.figure(figsize=(10,8))
            ax            = fig.add_subplot(111, projection='3d')
            scatter       = ax.scatter(results[gray_mask,0], results[gray_mask,1], results[gray_mask,2], c='lightgray', label='Other patients', s=5, alpha=0.7)
                
            x_patient     = results[color_mask]
            norm          = Normalize(vmin=0, vmax= (labels == specified_patient).sum())
            n_epochs      = np.arange(x_patient.shape[0])
            sm            = ax.scatter(x_patient[:,0],x_patient[:,1],x_patient[:,2], c=n_epochs, cmap=cmap, norm=norm, label=f'Patient {specified_patient}', s=5, alpha=0.7)

            ax.set_title(f'{method_name} visualization')
            fig.colorbar(sm, ax=ax, label=f'Epochs of patient {specified_patient}')
            ax.legend()
            fig.savefig(f'{path}/{method_name}_plot_{specified_patient}.png')
            plt.close()

        if dimension_tsne == 2:
            fig           = plt.figure(figsize=(10,8))
            scatter       = plt.scatter(results[gray_mask,0], results[gray_mask,1], c='lightgray', label='Other patients', s=5, alpha=0.7)
                
            x_patient     = results[color_mask]
            norm          = Normalize(vmin=0, vmax= (labels == specified_patient).sum())
            cmap          = plt.cm.plasma
            n_epochs      = np.arange(x_patient.shape[0])
            sm            = plt.scatter(x_patient[:,0],x_patient[:,1], c=n_epochs, cmap=cmap, norm=norm, label=f'Patient {specified_patient}', s=5, alpha=0.7)

            plt.title(f'{method_name} visualization')
            plt.colorbar(sm, label=f'Epochs of patient {specified_patient}')
            plt.legend()
            fig.savefig(f'{path}/{method_name}_plot_{specified_patient}.png')
            plt.close()
        


