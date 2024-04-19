# Biogears-ml

## Generating data
The simulated patient data can be generated by running `python generate-asthma-dataset.py`. The DIR_BIOGEARS_BIN variable in global_constants.py should be changed to the location of the BioGears executable. Our generated data can be found in the `asthma-dataset-1` folder.

## Visualizing data
The patient health metrics over time can be visualized by running `python graph-csv-1.py`, which outputs plots for the behaviour of a patient with life-threatening asthma and a patient without any asthma conditions.

## Data Preprocessing
The augmented data can be generated by running `python preprocess-dataset.py`. This creates copies of the original data each with Gaussian noise added in the `asthma-dataset-1-augmented-1` folder.

## Model Training
The model can be trained and evaluated by running `python train-model.py`. This contains the main functions and outputs training progress with plotted results. 
