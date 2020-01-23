import os
import pickle

os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,force_device=True"
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
import pandas as pd
import pymc3 as pm
from sklearn.metrics import r2_score
from configuration import HIERARCHICAL_MODEL_PERFORMANCE_FOLDER_2_LEVELS_2_COVARIATE, HIERARCHICAL_MODEL_INFERENCE_FOLDER_2_LEVELS_2_COVARIATE, \
    DATA_TRAINING_FILE, DATA_TESTING_FILE, CONFIG_FILE
from models.hierarchical_2_levels_2_covariate.performance_check import do_prediction, calc_accurracy, input_data



def main(output_trace_path, Xy_training_path, Xy_testing_path, output_path, main_cities):
    # loading data
    with open(output_trace_path, 'rb') as buff:
        data = pickle.load(buff)
        hierarchical_model, hierarchical_trace, scaler, degree_index, \
        response_variable, predictor_variables = data['inference'], data['trace'], data['scaler'], \
                                                 data['city_index_df'], data['response_variable'], \
                                                 data['predictor_variables']

    # fields to scale, get data of traces
    fields_to_scale = [response_variable] + predictor_variables
    Xy_testing, Xy_training = input_data(Xy_testing_path, Xy_training_path, fields_to_scale, scaler)

    # get data of traces and only 1000 random samples
    data = pm.trace_to_dataframe(hierarchical_trace)
    data = data.sample(n=1000).reset_index(drop=True)

    index_climatezone = degree_index.pivot_table(index=['CLIMATE_ZONE'])

    alpha_training = []
    beta_training = []
    gamma_training = []
    num_buildings_train = Xy_training.shape[0]
    for building in range(num_buildings_train):
        buildig_climate_zone = Xy_training.loc[building, "CLIMATE_ZONE"]
        index = index_climatezone[buildig_climate_zone].index_d.values[0]
        alpha_training.append(data['degree_state_b__' + str(index)].mean())
        beta_training.append(data['degree_state_m__' + str(index)].mean())
        gamma_training.append(data['degree_state_g__' + str(index)].mean())

    alpha_testing = []
    beta_testing = []
    gamma_testing = []
    num_buildings_test = Xy_testing.shape[0]
    for building in range(num_buildings_test):
        buildig_climate_zone = Xy_testing.loc[building, "CLIMATE_ZONE"]
        index = index_climatezone[buildig_climate_zone].index_d.values[0]
        alpha_testing.append(data['degree_state_b__' + str(index)].mean())
        beta_testing.append(data['degree_state_m__' + str(index)].mean())
        gamma_testing.append(data['degree_state_g__' + str(index)].mean())

    # do for the training data set
    Xy_training["prediction"], Xy_training["observed"], _, _ = do_prediction(Xy_training, alpha_training, beta_training,
                                                                             gamma_training,
                                                                             response_variable,
                                                                             predictor_variables,
                                                                             fields_to_scale,
                                                                             scaler)
    # do for the testing data set
    Xy_testing["prediction"], Xy_testing["observed"], _, _ = do_prediction(Xy_testing, alpha_testing, beta_testing,
                                                                           gamma_testing,
                                                                           response_variable,
                                                                           predictor_variables,
                                                                           fields_to_scale,
                                                                           scaler)
    climate_zones = set(index_climatezone.index)
    accurracy_df = pd.DataFrame()
    for climate_zone in climate_zones:
        temporal_data_train = Xy_training[Xy_training["CLIMATE_ZONE"] == climate_zone]
        temporal_data_test = Xy_testing[Xy_testing["CLIMATE_ZONE"] == climate_zone]

        if temporal_data_train.empty or temporal_data_test.empty:
            print(climate_zone, "does not exist, we are skipping it")
        else:
            n_samples_train = temporal_data_train.shape[0]
            n_samples_test = temporal_data_test.shape[0]

            MAPE_single_building_train, MAPE_city_scale_train, r2_test = calc_accurracy(
                np.array(temporal_data_train["prediction"]),
                np.array(temporal_data_train["observed"]))
            MAPE_single_building_test, MAPE_city_scale_test, r2_test = calc_accurracy(
                np.array(temporal_data_test["prediction"]),
                np.array(temporal_data_test["observed"]))

            dictionary = pd.DataFrame.from_items([("CLIMATE_ZONE", [climate_zone, climate_zone, ]),
                                            ("DATASET", ["Training", "Testing"]),
                                            ("MAPE_%",
                                             [MAPE_single_building_train, MAPE_single_building_test]),
                                            ("PE_%", [MAPE_city_scale_train, MAPE_city_scale_test]),
                                            ("n_samples", [n_samples_train, n_samples_test])])

            accurracy_df = pd.concat([accurracy_df, dictionary], ignore_index=True)
    # append both datasets
    accurracy_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    name_model = "log_log_all_2var_standard_5000"
    output_path = os.path.join(HIERARCHICAL_MODEL_PERFORMANCE_FOLDER_2_LEVELS_2_COVARIATE, name_model + "CLIMATE_ZONE.csv")
    output_trace_path = os.path.join(HIERARCHICAL_MODEL_INFERENCE_FOLDER_2_LEVELS_2_COVARIATE, name_model + ".pkl")
    Xy_training_path = DATA_TRAINING_FILE
    Xy_testing_path = DATA_TESTING_FILE
    main_cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values

    main(output_trace_path, Xy_training_path, Xy_testing_path, output_path, main_cities)
