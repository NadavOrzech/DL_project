# CS236781: Deep Learning on Computational Accelerators
# Final Project - Detect Atrial Fibrillation using long short-term memory networks (LSTM) with RR interval signals
# Nadav Orzech & Roni Englender

## Structure of the code:

- config.py - Module which holds the class with hyperparameters for the entire project code.

- data_preprocess.py - Module which holds all the preprocess methods of the raw data and generate the relevant datasets. 
					   The main class here is DataProcessor which splits the data into beats and beat sequences and generates datasets according to the config class. 

- data_loader.py - Module which generates the train and test dataloaders for the project models.

- model.py - Holds the project's models: 
					* BaselineModel class - model which implements the baseline paper approach.
					* AttentionModel class - model which derives from BaselineModel class and implements our approach and improvements.

- cs236781/train_results - Module which holds result classes for easy saving and passing results from one class to another.
						   This module taken from the course homework assignments and modyfied by us.

- cs236781/plot.py - Module which the 3 graphs generation functions:
						* plot_fit - function which generates graph showing the progress of the model durining training and testing.
						* plot_both_models - function which generates graph comparing between the baseline model and our model.
						* plot_attention_map - function which generates graph showing our attention layer whights corresponding to the ground truth data.
				     This module taken from the course homework assignments and modyfied by us.

- main.py - Module which runs the project expirements and generates the the presented graphs.

- environment.yml - is the conda environment file

## How to reproduce the experiments

- To configure the conda environment you can use the environment.yml file
- Download and unzip the MIT-BIH Atrial Fibrillation Database from physionet (https://physionet.org/content/afdb/1.0.0/)
- Change in the config file the field of 'files_dir' to the unzipped directory.
- Run the main.py file to reproduce the experiments with the hyperparameters within the config file. 
  This will generates the models checkpoint files in 'checkpoint' directory and dataset checkpoint files in 'dataset_checkpoints' directory.
- In the end of the run 3 new files will be created displaying the graphs ('Attention_graph.png', 'Baseline_graph.png', 'compare.png').
  Also  a new directory will be created by the name 'heatmaps' which will hold several examples of the attention map graphs.
Note: the graphs will be created only on a 'cpu' machine. if you run this code on a 'cuda' machine you will need to copy the checkpoint files (checkpoints and dataset_checkpoints) to a 'cpu' machine and run the main.py again to generate the graphs (the model training will not run again).