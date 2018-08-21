import os

#GENERAL STUFF

#indicate relative or absolute path to your configuration.xlsx file
CONFIG_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "configuration.xlsx")

#indicate relative or absolute path to training, testing and prediction datasets and folders
DATA_TRAINING_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases", "training_and_testing", "training_database.csv")
DATA_TESTING_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases", "training_and_testing", "testing_database.csv")
DATA_ALLDATA_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases", "training_and_testing", "all_database.csv")
DATA_PREDICTION_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases", "prediction")


##ADVANCED STUFF
#indicate relative or absolute path to your data_raw folders (ONLY FOR PRE-PROCESSING)
DATA_RAW_BUILDING_GEOMETRY_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data_raw", "building_geometry", "data")
DATA_RAW_BUILDING_PERFORMANCE_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data_raw", "building_performance", "data_300k")
DATA_RAW_BUILDING_IPCC_SCENARIOS_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data_raw", "IPCC_SCENARIOS", "data", "future_degree_days.csv")
DATA_RAW_BUILDING_TODAY_HDD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data_raw", "today_heating_degree_days", "data")

#indicate folders to store the results (I ADVISE TO KEEP THEM AS IT IS)
HIERARCHICAL_MODEL_INFERENCE_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "results", "hierarchical", "inference")
HIERARCHICAL_MODEL_PERFORMANCE_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "results", "hierarchical", "performance")
HIERARCHICAL_MODEL_PREDICTION_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "results", "hierarchical", "predictions")
HIERARCHICAL_MODEL_COEFFICIENT_PLOTS_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "analysis", "plots", "hierarchical")
DATA_ANALYSIS_PLOTS_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "analysis", "plots", "other_analysis")
DATA_HDD_CDD_PLOTS_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "analysis", "plots", "IPCC_scenarios")
