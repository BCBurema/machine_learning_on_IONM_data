'''
This module trains an encoder using the TS2Vec framework on preprocessed motor evoked potentials. 

Usage:
    python train.py run_name dataset --...
    All arguments that can be used are written below in the parser.

Author: B.C. Burema
Date: 21/09/2026
'''

###------IMPORTING LIBRARIES------###
import numpy as np
import pandas as pd
import argparse
import os
import time
from ts2vec import TS2Vec
from utils import init_dl_program, name_with_datetime
from visualization_cml import tsne_plot
from tasks import alarm_criteria
from tasks.ionm_dashboard import create_app
import umap.umap_ as umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

###------CREATE PARSER -> TRAIN TS2VEC ------###
def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

if __name__ == '__main__':
    def str_or_int(value):
        try:
            return int(value)
        except ValueError:
            return value

    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('dataset', help='Which dataset to train on. Choose from: left_leg, right_leg, left_foot, right_foot, leg, foot, test')
    parser.add_argument('--gpu', type=str_or_int, default='cpu', help='The gpu no. used for training and inference (defaults to 0) (for cpu do cpu, for gpu do 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 16)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--hidden-dims', type=int, default=64, help='The hidden dimension of the encoder (defaults to 64)')
    parser.add_argument('--depth', type=int, default=9, help='The number of hidden residual blocks in the encoder (defaults to 9)')
    parser.add_argument('--max-train-length', type=int, default=None, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--temporal-unit', type=int, default=2, help='The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=10, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=26, help='The random seed') 
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--scheduler', action='store_true', help='Whether to use the learning rate scheduler')
    parser.add_argument('--notrain', action="store_true", help='Whether to skip training of the encoder, loading existing model')
    parser.add_argument('--nosave', action="store_true", help='Whether to skip saving the model and output')
    parser.add_argument('--visualization', action="store_true",help='Whether to plot tsne or UMAP')
    parser.add_argument('--alarm_criteria',action="store_true",help='Whether to calculate alarm features based on embedded space')
    parser.add_argument('--dashboard', action="store_true", help='Whether to create dashboard of alarm criteria')
    args = parser.parse_args()

    # Paths
    BASE_DIR          = './output/'                                 # Define your own folder here
    data_encoder_path = BASE_DIR + r'\data_encoder'
    save_path         = BASE_DIR + r'\results_contrastive_learning'
    alarm_path        = os.path.join(save_path, 'alarming')

    data_paths        = {
        'leg': {
            'X':                         os.path.join(data_encoder_path, 'X_leg.npy'),
            'epoch_ids':                 os.path.join(data_encoder_path, 'epoch_ids_leg.npy'),
            'patient_labels':            os.path.join(data_encoder_path, 'epoch_patient_labels_leg.npy'),
            'outcome_labels':            os.path.join(data_encoder_path, 'epoch_outcome_labels_leg.npy'),
            'side_labels':               os.path.join(data_encoder_path,'epoch_side_labels_leg.npy'),
            'center_labels':             os.path.join(data_encoder_path, 'epoch_center_labels_leg.npy'),
            'surgical_duration':         os.path.join(data_encoder_path,'epoch_surgical_duration_leg.npy'),
            'abs_energy':                os.path.join(data_encoder_path, 'abs_energy_leg.npy'),
            'abs_max':                   os.path.join(data_encoder_path, 'abs_max_leg.npy'),
            'amplitude':                 os.path.join(data_encoder_path, 'amplitude_leg.npy'),
            'latency':                   os.path.join(data_encoder_path, 'latency_leg.npy'),
            'energy_ratio_6':            os.path.join(data_encoder_path, 'energy_ratio_focus_6.npy'),
            'energy_ratio_7':            os.path.join(data_encoder_path, 'energy_ratio_focus_7.npy'),
            'energy_ratio_8':            os.path.join(data_encoder_path, 'energy_ratio_focus_8.npy'),
            'energy_ratio_9':            os.path.join(data_encoder_path, 'energy_ratio_focus_9.npy'),
            'kurtosis_energy_ratio_6':   os.path.join(data_encoder_path, 'kurt_energy_ratio_focus_6.npy'),
            'kurtosis_energy_ratio_7':   os.path.join(data_encoder_path, 'kurt_energy_ratio_focus_7.npy'),
            'kurtosis_energy_ratio_8':   os.path.join(data_encoder_path, 'kurt_energy_ratio_focus_8.npy'),
            'kurtosis_energy_ratio_9':   os.path.join(data_encoder_path, 'kurt_energy_ratio_focus_9.npy'),
            'length':                    os.path.join(data_encoder_path, 'length_leg.npy'),
            'alarm_features':            os.path.join(alarm_path, 'alarm_features_leg.csv'),
            'statistical_results':       os.path.join(alarm_path, 'statistical_results_leg.csv'),
            'statistical_results_left':  os.path.join(alarm_path, 'statistical_results_leg_left.csv'),
            'statistical_results_right': os.path.join(alarm_path, 'statistical_results_leg_right.csv'),
            'mep_alarm_timeseries':      os.path.join(alarm_path, 'mep_alarm_timeseries_leg.csv'),
            'master_df':                 os.path.join(alarm_path, 'master_df_leg.csv')
        },
        'foot': {
            'X':                         os.path.join(data_encoder_path, 'X_foot.npy'),
            'epoch_ids':                 os.path.join(data_encoder_path, 'epoch_ids_foot.npy'),
            'patient_labels':            os.path.join(data_encoder_path, 'epoch_patient_labels_foot.npy'),
            'outcome_labels':            os.path.join(data_encoder_path, 'epoch_outcome_labels_foot.npy'),
            'side_labels':               os.path.join(data_encoder_path,'epoch_side_labels_foot.npy'),
            'center_labels':             os.path.join(data_encoder_path, 'epoch_center_labels_foot.npy'),
            'surgical_duration':         os.path.join(data_encoder_path,'epoch_surgical_duration_foot.npy'),
            'alarm_features':            os.path.join(alarm_path, 'alarm_features_foot.csv'),
            'statistical_results':       os.path.join(alarm_path, 'statistical_results_foot.csv'),
            'statistical_results_left':  os.path.join(alarm_path, 'statistical_results_foot_left.csv'),
            'statistical_results_right': os.path.join(alarm_path, 'statistical_results_foot_right.csv'),
            'mep_alarm_timeseries':      os.path.join(alarm_path, 'mep_alarm_timeseries_foot.csv'),
            'master_df':                 os.path.join(alarm_path, 'master_df_foot.csv')
        },
    }

    paths       = data_paths[args.dataset]
    
    # Choose whether to use UMAP or not: if set to False, t-SNE visualization is applied
    use_umap                = True
    dimension_visualization = 2

    # Choose type of visualization: if set to False, that type of coloring is not applied
    specified_patient_true_false = False                        # Set to True if gradient coloring per patient is desired
    all_patients_true_false      = False                        # Set to True if each patient should be assigned a unique color
    outcome_labels_true_false    = False                        # Set to True if coloring on outcome is desired       
    side_labels_true_false       = False                        # Set to True if coloring on left and right side is desired
    surgical_duration_plot       = False                        # Set to True if coloring on surgical duration is desired
    feature_labels               = True                         # Set to True if coloring on feature_labels is desired -> only possible for leg
    center_labels_true_false     = False                        # Set to True if coloring on center is desired: GOSH vs EMC

    print("Arguments:", str(args))
    
    # Create the run directory to save the model and output
    if not args.nosave:
        run_dir = os.path.join(save_path, 'training', name_with_datetime(args.run_name))
        os.makedirs(save_path, exist_ok = True)
        os.makedirs(run_dir, exist_ok = True)

    ### Visualization ###
    # Train TS2VEc on complete dataset + visualize complete representation space
    if args.visualization or args.alarm_criteria:
        # Retrieve paths
        X_full              = np.load(paths['X'])
        patient_labels      = np.load(paths['patient_labels'], allow_pickle=True)
        outcome_labels      = np.load(paths['outcome_labels'], allow_pickle=True)
        side_labels         = np.load(paths['side_labels'], allow_pickle=True)
        center_labels       = np.load(paths['center_labels'],allow_pickle=True)
        epoch_ids           = np.load(paths['epoch_ids'],allow_pickle=True)
        surgical_durations  = np.load(paths['surgical_duration'],allow_pickle=True)

        if args.dataset == 'leg':
            # Retrieve features
            abs_energy          = np.load(paths['abs_energy'],allow_pickle=True)
            abs_max             = np.load(paths['abs_max'],allow_pickle=True)
            amplitude           = np.load(paths['amplitude'],allow_pickle=True)
            latency             = np.load(paths['latency'],allow_pickle=True)
            energy_ratio_6      = np.load(paths['energy_ratio_6'],allow_pickle=True)
            energy_ratio_7      = np.load(paths['energy_ratio_7'],allow_pickle=True)
            energy_ratio_8      = np.load(paths['energy_ratio_8'],allow_pickle=True)
            energy_ratio_9      = np.load(paths['energy_ratio_9'],allow_pickle=True)
            kurt_energy_ratio_6 = np.load(paths['kurtosis_energy_ratio_6'],allow_pickle=True)
            kurt_energy_ratio_7 = np.load(paths['kurtosis_energy_ratio_7'],allow_pickle=True)
            kurt_energy_ratio_8 = np.load(paths['kurtosis_energy_ratio_8'],allow_pickle=True)
            kurt_energy_ratio_9 = np.load(paths['kurtosis_energy_ratio_9'],allow_pickle=True)
            length              = np.load(paths['length'],allow_pickle=True)

        # Train encoder
        device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

        config = dict(
            batch_size       = args.batch_size,
            lr               = args.lr,
            output_dims      = args.repr_dims,
            hidden_dims      = args.hidden_dims,
            depth            = args.depth,
            max_train_length = args.max_train_length,
            temporal_unit    = args.temporal_unit,
        )

        # Initialize the model
        model = TS2Vec(
            input_dims = X_full.shape[-1],        # number of channels
            device     = device,
            **config
        )

        # Load a model if --notrain is set
        if args.notrain:
            # Specify model to load here
            print('Skipping training of encoder, loading existing model')
            if args.dataset == 'leg':
                repr = np.load(f'{save_path}/training/leg_final/repr.npy')
            else:
                repr = np.load(f'{save_path}/training/foot_final/repr.npy')
        else:
            # Train the model here
            loss_log, optimizer = model.fit(
                X_full,
                use_scheduler   = args.scheduler,
                n_epochs        = args.epochs,
                n_iters         = args.iters,
                verbose         = True
            )
            
            if not args.nosave:
                model.save(f'{run_dir}/model_full_tsne_seed_{args.seed}.pkl', optimizer=optimizer)

                repr = model.encode(X_full, encoding_window='full_series' if patient_labels.ndim == 1 else None) 

                np.save(os.path.join(run_dir,'repr.npy'),repr)

        if args.visualization:
            print('Create t-SNE or UMAP plots')
            if all_patients_true_false: 
                # Each patient is assigned an unique color
                tsne_plot(
                    run_dir,
                    repr,
                    patient_labels,
                    dimension_visualization,
                    args.seed,
                    all_patients_true_false,
                    use_umap=use_umap
                )

            if outcome_labels_true_false:
                # Epochs are colored on outcome
                tsne_plot(
                    run_dir,
                    repr,
                    patient_labels,
                    dimension_visualization,
                    args.seed,
                    outcome_labels = outcome_labels, 
                    use_umap=use_umap
                )

            if side_labels_true_false:
                # Epochs are colored on side
                tsne_plot(
                    run_dir,
                    repr,
                    patient_labels,
                    dimension_visualization,
                    args.seed,
                    side_labels = side_labels,
                    use_umap=use_umap 
                )
            
            if center_labels_true_false:
                # Epochs are colored on center
                tsne_plot(
                    run_dir,
                    repr,
                    patient_labels,
                    dimension_visualization,
                    args.seed,
                    center_labels = center_labels,
                    use_umap=use_umap 
                )

            if surgical_duration_plot:
                # Epochs are colored gradiently on surgical duration
                tsne_plot(
                    run_dir,
                    repr,
                    patient_labels,
                    dimension_visualization,
                    args.seed,
                    surgical_durations = surgical_durations,
                    outcome_labels = outcome_labels,
                    use_umap=use_umap 
                )

            if feature_labels:
                # Epochs are colored on features retrieved from preprocessed MEPs using TSFRESH
                features_to_plot = [
                    (amplitude, 'Amplitude'),
                    (latency, 'Latency'),
                    (energy_ratio_6, 'Energy ratio focus segment 7'),
                    (energy_ratio_7, 'Energy ratio focus segment 8'),
                    (energy_ratio_8, 'Energy ratio focus segment 9'),
                    (energy_ratio_9, 'Energy ratio focus segment 10'),
                    (kurt_energy_ratio_6, 'Energy ratio focus segment 7'),
                    (kurt_energy_ratio_7, 'Energy ratio focus segment 8'),
                    (kurt_energy_ratio_8, 'Energy ratio focus segment 9'),
                    (kurt_energy_ratio_9, 'Energy ratio focus segment 10'),
                    (length, 'Length time series')
                ]   

                for feature_label, feature_name in features_to_plot:
                    tsne_plot(
                        run_dir,
                        repr,
                        patient_labels,
                        dimension_visualization,
                        args.seed,
                        feature_labels=feature_label,
                        feature_name=feature_name,
                        use_umap=use_umap 
                    )

            if specified_patient_true_false: 
                # All patients are gray, except for the specified patient -> gradient colored.
                for patient in np.unique(patient_labels):
                    tsne_plot(
                        run_dir,
                        repr,
                        patient_labels,
                        dimension_visualization,
                        args.seed,
                        specified_patient = patient
                    )
        
        if args.alarm_criteria:
            if not os.path.exists(paths['alarm_features']):
                print('Calculate summary alarm features -> one value per metric per extremity')
                alarm_features  = alarm_criteria.compute_alarm_features(repr, patient_labels, side_labels, outcome_labels, epoch_ids)
                alarm_features.to_csv(paths['alarm_features'], index=False)
            else:
                alarm_features  = pd.read_csv(paths['alarm_features'])
                pd.set_option('display.max_columns',None)
                feats           = ['auc_baseline_dist_euclidean',
                                    'auc_baseline_dist_manhattan',
                                    'auc_baseline_dot',
                                    'auc_consecutive_dist_euclidean',
                                    'auc_consecutive_dist_manhattan',
                                    'auc_consecutive_dot',
                                    'net_displacement']
                
                print('Perform statistical analyses')
                results_all     = alarm_criteria.run_alarm_statistics(alarm_features, feats, side=None)
                results_all.to_csv(paths['statistical_results'],index=False)

                ## Uncomment this if you'd like to perform statistical analyses per side
                # results_left  = alarm_criteria.run_alarm_statistics(alarm_features, feats, side='left')
                # results_right = alarm_criteria.run_alarm_statistics(alarm_features, feats, side='right')
                # results_left.to_csv(paths['statistical_results_left'],index=False)
                # results_right.to_csv(paths['statistical_results_right'],index=False)

                # Create a boxplot of the alarm criteria
                feature_labels = {
                    'auc_baseline_dist_euclidean':      'AUC baseline dist. (Euclidean)',
                    'auc_baseline_dist_manhattan':      'AUC baseline dist. (Manhattan)',
                    'auc_baseline_dot':                 'AUC baseline dot product',
                    'auc_consecutive_dist_euclidean':   'AUC consecutive dist. (Euclidean)',
                    'auc_consecutive_dist_manhattan':   'AUC consecutive dist. (Manhattan)',
                    'auc_consecutive_dot':              'AUC consecutive dot product',
                    'net_displacement':                 'Net displacement'
                }

                group0      = alarm_features[alarm_features['outcome']==0]
                group1      = alarm_features[alarm_features['outcome']==1]

                fig, axes = plt.subplots(2,4, figsize=(16,8))
                axes      = axes.flatten()

                for idx, (feat, label) in enumerate(feature_labels.items()):
                    ax = axes[idx]
                    ax.boxplot(
                        [group0[feat].values, group1[feat].values],
                        labels=['No decline','Motor decline'],
                        patch_artist=True,
                        boxprops=dict(facecolor='lightblue',color='navy'),
                        medianprops=dict(color='red',linewidth=1),
                        whiskerprops=dict(color='navy'),
                        capprops=dict(color='navy'),
                        flierprops=dict(marker='o',markerfacecolor='gray',markersize=3,alpha=0.5)
                    )
                    ax.set_title(label, fontsize=10)
                    ax.set_ylabel('Value',fontsize=8)
                
                plt.suptitle('Alarm features: no decline vs motor decline',fontsize=13,fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir,'boxplot_alarm_features.png'),dpi=300,bbox_inches='tight')
                plt.close()

            if not os.path.exists(paths['mep_alarm_timeseries']):
                print('Calculate alarm criteria per MEP throughout surgery -> one value per MEP')
                alarm_features_meps = alarm_criteria.compute_mep_alarm_timeseries(repr, patient_labels, side_labels, outcome_labels, epoch_ids)
                alarm_features_meps.to_csv(paths['mep_alarm_timeseries'],index=False)

            if not os.path.exists(paths['master_df']):
                # Save UMAP information for the IONM Alarm dashboard
                umap = umap.UMAP(
                    n_components=dimension_visualization,
                    n_neighbors=30,
                    min_dist=0.3,
                    metric='euclidean',
                    random_state=args.seed,
                    n_jobs=1,   
                    verbose=True,
                    init='random',
                    n_epochs=500
                )
                umap_coords = umap.fit_transform(repr)

                master_df   = pd.DataFrame({
                    'umap_x':       umap_coords[:,0].astype(float),
                    'umap_y':       umap_coords[:,1].astype(float),
                    'patient_id':   patient_labels.astype(str),
                    'side':         side_labels.astype(str),
                    'outcome':      outcome_labels.astype(int),
                    'epoch_id':     epoch_ids.astype(str),
                    'epoch_index':  np.arange(len(epoch_ids)).astype(int),
                })
                master_df.to_csv(paths['master_df'], index=False)
    
    if args.dashboard:
        # Create dashboard
        # Also retrieve preprocessed MEPs to visualize the timeseries within the IONM Alarm Dashboard
        preprocessed_data_path  = BASE_DIR + r'\preprocessed_data'
        alarm_path              = os.path.join(save_path, 'alarming')
        app                     = create_app(alarm_path, preprocessed_data_path, args.dataset)
        print('IONM alarm dashboard on http://localhost:8050')
        app.run(debug=False, port=8050) # use_reloader=False
    print("Finished.")
