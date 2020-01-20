import os

#GENERAL STUFF

#indicate relative or absolute path to your configuration.xlsx file
CONFIG_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "configuration.xlsx")

#indicate relative or absolute path to training, testing and prediction datasets and folders
DATA_TRAINING_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases", "training_and_testing", "training_database.csv")
DATA_TESTING_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases", "training_and_testing", "testing_database.csv")
DATA_ALLDATA_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases", "training_and_testing", "all_database.csv")
DATA_PREDICTION_FOLDER_TODAY_EFFICIENCY = os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases", "prediction_today_efficiency")
DATA_PREDICTION_FOLDER_FUTURE_EFFICIENCY = os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases", "prediction_future_efficiency")
DATA_TODAY_CONSUMPTION_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases", "today_energy_consumption", "consumption.csv")
DATA_FUTURE_EFFICIENCY_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases", "today_future_efficiency", "building_efficiency.xlsx")


##ADVANCED STUFF
#indicate relative or absolute path to your data_raw folders (ONLY FOR PRE-PROCESSING)
DATA_RAW_BUILDING_GEOMETRY_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data_raw", "building_geometry", "data")
DATA_RAW_BUILDING_PERFORMANCE_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data_raw", "building_performance", "data_300k")
DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data_raw", "IPCC_SCENARIOS")
DATA_RAW_BUILDING_IPCC_SCENARIOS_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases", "future_degree_days", "future_degree_days.csv")
DATA_RAW_BUILDING_ENTHALPY_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases", "enthalpy_growth")
DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases", "today_future_enthalpy", "future_enthalpy_days.csv")
DATA_OPTIMAL_TEMPERATURE_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases", "training_and_testing", "temperature_optimal.csv")
DATA_RAW_BUILDING_TODAY_HDD_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data_raw", "today_heating_degree_days", "data")

DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE_EFFICEINCY = os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases", "today_future_enthalpy", "future_enthalpy_days_efficiency.csv")

#indicate folders to store the results (I ADVISE TO KEEP THEM AS IT IS)
HIERARCHICAL_MODEL_INFERENCE_FOLDER_2_LEVELS = os.path.join(os.path.abspath(os.path.dirname(__file__)), "results", "hierarchical_2_levels", "inference")
HIERARCHICAL_MODEL_PERFORMANCE_FOLDER_2_LEVELS = os.path.join(os.path.abspath(os.path.dirname(__file__)), "results", "hierarchical_2_levels", "performance")
HIERARCHICAL_MODEL_PREDICTION_FOLDER_2_LEVELS = os.path.join(os.path.abspath(os.path.dirname(__file__)), "results", "hierarchical_2_levels", "predictions")
HIERARCHICAL_MODEL_COEFFICIENT_PLOTS_FOLDER_2_LEVELS = os.path.join(os.path.abspath(os.path.dirname(__file__)), "analysis", "plots", "hierarchical_2_levels")

HIERARCHICAL_MODEL_INFERENCE_FOLDER_2_LEVELS_2_COVARIATE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "results", "hierarchical_2_levels_2_covariate", "inference")
HIERARCHICAL_MODEL_PERFORMANCE_FOLDER_2_LEVELS_2_COVARIATE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "results", "hierarchical_2_levels_2_covariate", "performance")
HIERARCHICAL_MODEL_PREDICTION_FOLDER_2_LEVELS_2_COVARIATE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "results", "hierarchical_2_levels_2_covariate", "predictions")
HIERARCHICAL_MODEL_COEFFICIENT_FOLDER_2_LEVELS_2_COVARIATE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "analysis", "plots", "hierarchical_2_levels_2_covariate")


HIERARCHICAL_MODEL_INFERENCE_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "results", "hierarchical", "inference")
NN_MODEL_INFERENCE_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "results", "neural_network", "inference")
NN_MODEL_PERFORMANCE_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "results", "neural_network", "performance")
NN_MODEL_PREDICTION_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "results", "neural_network", "predictions")
HIERARCHICAL_MODEL_PERFORMANCE_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "results", "hierarchical", "performance")
HIERARCHICAL_MODEL_PREDICTION_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "results", "hierarchical", "predictions")
HIERARCHICAL_MODEL_COEFFICIENT_PLOTS_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "analysis", "plots", "hierarchical")
DATA_ANALYSIS_PLOTS_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "analysis", "plots", "other_analysis")
DATA_ENTHALPY_GROWTH_PLOTS_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "analysis", "plots", "enthalpy_growth")
DATA_HDD_CDD_PLOTS_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "analysis", "plots", "IPCC_scenarios")
DATA_ENERGY_PLOTS_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "analysis", "plots", "energy")
DATA_CLUSTERING_PLOTS_FOLDER  = os.path.join(os.path.abspath(os.path.dirname(__file__)), "analysis", "plots", "clustering")

DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER = os.path.join(os.path.abspath(os.path.dirname(__file__)), "databases", "future_energy_consumption")
DATA_POST_FUTURE_ENERGY_CONSUMPTION_FILE = os.path.join(DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER, "future_consumption.csv")
DATA_POST_FUTURE_ENERGY_CONSUMPTION_FILE_WITH_EFFICIENCY = os.path.join(DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER, "future_consumption_efficiency.csv")

ZONE_NAMES = {"Hot-humid":["1A", "2A", "3A"] ,
               "Hot-dry": ["2B", "3B"],
               "Hot-marine": ["3C"] ,
               "Mixed-humid": ["4A"],
               "Mixed-dry": ["4B"],
               "Mixed-marine": ["4C"],
               "Cold-humid": ["5A", "6A"],
               "Cold-dry": ["5B", "6B", "7"]}



