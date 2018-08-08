# great-american-cities
paper-american-cities

## Installation for the Hierarchical Bayesian model
1. install anaconda distribution for python 3 and python 3
2. install pymc3, scikitlearn and pandas. Probably you need to install more, so python will let you know in case we are missing it.

## Installation for the Neural Network
1. install tensorflow, I advise to use a GPU to run it quickly.

## Step 1. Configure the script
a. open the configuration.xlsx/test_cities and indicate the names of the cities to evaluate.
b. open the configuration.py and indicate the paths to the datasets and the configuration.xlsx.
   do this estep only if you have an alternative database to that one provided in the repository.

## Step 2. Define and run the hierachical model
a. Open models/hierarchical/1_definition_and_inference.py
b. check the input configurations at the end of the script
c. run the script

## Step 3. Check performance of the hierachical model
a. Open models/hierarchical/2_performance_check.py
b. Indicate the name of the model previously run to check
c. run the script

## Step 4. run some predictions of the hierachical model
a. Open models/hierarchical/3_predictions.py
b. Indicate the name of the model previously run to check
c. run the script

## Step 5. Check results in the results folder
a. Open results/hierarchical/inference/ [model name].pkl
b. Open results/hierarchical/performance/ [model name].csv
c. Open results/hierarchical/predictions/ [model name] / [city].csv

## Step 6. Also create some graphs about the results
a. Open results/hierarchical/inference/ [model name].pkl
b. Open results/hierarchical/performance/ [model name].csv
c. Open results/hierarchical/predictions/ [model name] / [city].csv
