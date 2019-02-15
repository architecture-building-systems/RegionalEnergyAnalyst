# Prediciton model of energy consumption in 100 american cities.


This repository stores the statistical models of the project.
It Includes a sub-set of the original training and testing database.
It Includes a sub-set of the original database for predictions.
This should be enough for testing the approach.

## Repository contents
- databases: It includes databases for training, testing and doing predictions with the models.
- models: it includes scripts to define, carry out inference, check performance and make predictions with a hierarchical and a deep NN model.
- results: It stores the results of the inference, performance and prediction processes.
- analysis: It includes scripts to generate tables and plots for further analysis. (the tables and plots are stored in there)

## Installation for the Hierarchical Bayesian model
1. install anaconda distribution for python 3 and python 3
2. install pymc3, theano, scikitlearn and pandas. Probably you need to install more, so python will let you know in case we are missing it.

## Installation for the Neural Network
1. install tensorflow, I advise to use a GPU to run it quickly.

## Step0. preprocess the data (sorry only available for the authors)
1. run the script `data_processing/IPCC_scenarios_cleaner.py`
2. run the script `data_processing/enthalpy_calculation.py`
3. run the script `data_processing/split_enthalpy_by_period.py.py`
4. run the script `data_processing/training_and_testing_database.py`
5. run the script `data_processing/prediction_database.py`

## Step 1. Configure the script
- open the excel file `configuration.xlsx/test_cities` and indicate the names of the cities to evaluate.
- open the script `configuration.py` and indicate the paths to the datasets and the `configuration.xlsx`. Do this step only if you have an alternative database to that one provided in the repository.

## Step 2. Define and run the hierachical model
- Open the script `models/hierarchical/1_definition_and_inference.py`
- Check the input configurations at the end of the script. A name of the model will be infered from these inputs.
- Run the script
- The results are stored in `results/hierarchical/inference/ [model name].pkl`

## Step 3. Check performance of the hierachical model
- Open the script `models/hierarchical/2_performance_check.py`
- Indicate the name of the model to use (e.g., log_log_all_standard_1000)
- Run the script
- The results are stored in `results/hierarchical/performance/ [model name].csv`

## Step 4. run some predictions of the hierachical model
- Open the script `models/hierarchical/3_predictions.py`
- Indicate the name of the model to use (e.g., log_log_all_standard_1000)
- Run the script
- The results are stored in `results/hierarchical/predictions/ [model name]/[city].csv`

## Step 5. Create tables and plots for analysis
- Run the script `analysis/coefficients_hierarchical_model.py` to create a PDF of the regression coefficients inferred with the hierarchical model.
- Run the script `analysis/data_pair_plot.py` to create a pairplot of the input training or test database.
- Run the script `analysis/IPCC-scenario_plots.py` to create a box_plot of the heating degree days of all IPCC scenarios.
- Run the script `analysis/plots_predictions.py` to create a box_plot of the energy consumption predicted for the IPCC sceanrios.
NOTE: You must specify the hierarchical model name in the inputs at the end of each script.
