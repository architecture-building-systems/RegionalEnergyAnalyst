import os

#GENERAL STUFF

#indicate relative or absolute path to your configuration.xlsx file
CONFIG_FILE = os.path.join(os.getcwd(), "configuration.xlsx")

#indicate relative or absolute path to training, testing and prediction datasets and folders
DATA_TRAINING_FILE = os.path.join(os.getcwd(), "databases", "training_and_testing", "training_database.csv")
DATA_TESTING_FILE = os.path.join(os.getcwd(), "databases", "training_and_testing", "testing_database.csv")
DATA_PREDICTION_FOLDER = os.path.join(os.getcwd(), "databases", "prediction")


##ADVANCED STUFF
#indicate relative or absolute path to your data_raw folders (ONLY FOR PRE-PROCESSING)
DATA_RAW_BUILDING_GEOMETRY_FOLDER = os.path.join(os.getcwd(), "data_raw", "building_geometry", "data")
DATA_RAW_BUILDING_PERFORMANCE_FOLDER = os.path.join(os.getcwd(), "data_raw", "building_performance", "data_300k")
DATA_RAW_BUILDING_IPCC_SCENARIOS_FILE = os.path.join(os.getcwd(), "data_raw", "IPCC_SCENARIOS", "data", "future_degree_days.csv")
DATA_RAW_BUILDING_TODAY_HDD_FOLDER = os.path.join(os.getcwd(), "data_raw", "today_heating_degree_days", "data")

#indicate folders to store the results (I ADVISE TO KEEP THEM AS IT IS)
HIERARCHICAL_MODEL_INFERENCE_FOLDER = os.path.join(os.getcwd(), "results", "hierarchical", "inference")
HIERARCHICAL_MODEL_PERFORMANCE_FOLDER = os.path.join(os.getcwd(), "results", "hierarchical", "performance")
HIERARCHICAL_MODEL_PREDICTION_FOLDER = os.path.join(os.getcwd(), "results", "hierarchical", "predictions")
HIERARCHICAL_MODEL_COEFFICIENT_PLOTS_FOLDER = os.path.join(os.getcwd(), "analysis", "plots", "hierarchical")
