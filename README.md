# machine_learning_on_IONM_data
This repository contains the code for the master's thesis of B.C. Burema, titled:
_Towards Intraoperative Decision Support: A Machine Learning Approach for Intraoperative Neuromonitoring in Pediatric Spinal Lipoma Surgery_

Read the full thesis here: 

**Overview**
This project consists of two main parts
1. **Supervised machine learning** for predicting postoperative motor outcomes after spinal lipoma surgery;
2. **Contrastive learning** for establishing alarm criteria based on learned representations.

**Repository structure**
The repository consists of three folders.
**1. preprocessing**
   Contains the preprocessing pipeline that transforms raw motor evoked potentials (MEPs) into preprocessed MEPs. The resulting files can be used as input for both the supervised machine learning and contrastive learning approaches below.

**2. supervised_ml**
   Contains all code for the supervised machine learning approach.
   To run, execute only main_supervised.py.
   
   Required adjustments:
   - paths (lines 35-47): update the paths to match your local setup.
     
   Optional adjustments:
   - threshold (line 310): the threshold used to convert probability predictions into class labels.
   - anova_k (lines 311-312): the number of features to select.
   - classifier (309): the classifier to use.
     
**3. contrastive_ml**
   Contains all code for the contrastive learning approach.
   To run it, use the following command (replace --ARGS with the desired arguments): 
   
   python train.py run_name dataset --ARGS

   followed by the arguments that you want:
   run_name (str): name of the folder for saving the model, representations, and outputs (required)
   dataset (str): either 'leg' or 'foot' (required)

   --gpu (int or str): GPU no. used for training (use 'cpu' to run on CPU) (default: 'cpu')
   --batch-size (int): batch size used for training (default: 8)
   --lr (float): learning rate (default: 0.001)
   --repr_dims (int): representation dimension (default: 320)
   --depth (int): Number of hidden residual blocks in encoder (default: 9)
   --max_train_length (int): maximum sequence length, longer sequences are split (default: None)
   --temporal-unit (int) minimum unit to perform temporal contrast (default: 2)
   --iters (int): number of training iterations (default: None)
   --epochs (int): number of training epochs (default: 10)
   --save-every (int): save a checkpoint every N iterations/epochs (default: None)
   --seed (int): the random seed for reproducibility (default: 26)
   --max-threads (int): maximum number of threads used by this process (default: None)
   --scheduler (flag): enable learning rate scheduler (default: off)
   --notrain (flag): skip training, load existing model (default: off)
   --nosave (flag): skip saving the model, representations, and outputs (default: off)
   --visualization (flag): whether to plot the t-SNE or UMAP (default: off)
   --alarm_criteria (flag): whether to compute the alarm criteria based on the representations (default: off)
   --dashboard (flag): whether to create and show the IONM Alarm Dashboard (default: off)

   Required adjustments:
   - paths (lines 70-74,): update the paths to match your local setup.
     
   Optional adjustments:
   - visualization (lines 124-135): choose between t-SNE and UMAP, and select the type of coloring.
   


