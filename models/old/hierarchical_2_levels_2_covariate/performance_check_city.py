import os
import pickle

# os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,force_device=True"
# os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
import pandas as pd
import pymc3 as pm
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

    # x = pm.summary(hierarchical_trace)
    alpha_training = []
    beta_training = []
    gamma_training = []
    num_buildings_train = Xy_training.shape[0]
    for building in range(num_buildings_train):
        buildig_city = Xy_training.loc[building, "CITY"]
        index = degree_index[(degree_index["CITY"] == buildig_city)].index_ds.values[0]
        alpha_training.append(data['degree_state_b__' + str(index)].mean())
        beta_training.append(data['degree_state_m__' + str(index)].mean())
        gamma_training.append(data['degree_state_g__' + str(index)].mean())

    alpha_testing = []
    beta_testing = []
    gamma_testing = []
    num_buildings_test = Xy_testing.shape[0]
    for building in range(num_buildings_test):
        buildig_city = Xy_testing.loc[building, "CITY"]
        index = degree_index[(degree_index["CITY"] == buildig_city)].index_ds.values[0]
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

    cities = set(degree_index["CITY"].values)
    accurracy_df = pd.DataFrame()
    for city in cities:
        temporal_data_train = Xy_training[Xy_training["CITY"] == city]
        temporal_data_test = Xy_testing[Xy_testing["CITY"] == city]

        if temporal_data_train.empty or temporal_data_test.empty:
            print(city, "does not exist, we are skipping it")
        else:
            n_samples_train = temporal_data_train.shape[0]
            n_samples_test = temporal_data_test.shape[0]

            MAPE_single_building_train, MAPE_city_scale_train, r2_test = calc_accurracy(
                np.array(temporal_data_train["prediction"]),
                np.array(temporal_data_train["observed"]))
            MAPE_single_building_test, MAPE_city_scale_test, r2_test = calc_accurracy(
                np.array(temporal_data_test["prediction"]),
                np.array(temporal_data_test["observed"]))

            dictionary = pd.DataFrame.from_items([("CITY", [city, city, ]),
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
    output_path = os.path.join(HIERARCHICAL_MODEL_PERFORMANCE_FOLDER_2_LEVELS_2_COVARIATE, name_model + "CITY.csv")
    output_trace_path = os.path.join(HIERARCHICAL_MODEL_INFERENCE_FOLDER_2_LEVELS_2_COVARIATE, name_model + ".pkl")
    Xy_training_path = DATA_TRAINING_FILE
    Xy_testing_path = DATA_TESTING_FILE
    main_cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values

    main(output_trace_path, Xy_training_path, Xy_testing_path, output_path, main_cities)
