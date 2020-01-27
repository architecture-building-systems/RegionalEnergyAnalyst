import os
import pickle

os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,force_device=True"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import pandas as pd
import pymc3 as pm
from configuration import CONFIG_FILE, \
    HIERARCHICAL_MODEL_PREDICTION_FOLDER_2_LEVELS_2_COVARIATE, HIERARCHICAL_MODEL_INFERENCE_FOLDER_2_LEVELS_2_COVARIATE, \
    DATA_PREDICTION_FOLDER_TODAY_EFFICIENCY


def input_data(Xy_prediction_path, response_variable, fields_to_scale, scaler):
    # READ DATA
    X_prediction = pd.read_csv(Xy_prediction_path)
    X_prediction[response_variable] = 0.0  # create dummy of response variable so we can scale all the data

    if scaler != None:
        X_prediction[fields_to_scale] = pd.DataFrame(scaler.transform(X_prediction[fields_to_scale]),
                                                     columns=X_prediction[fields_to_scale].columns)

    return X_prediction


def do_prediction(Xy_predict, alpha, beta, gamma, response_variable, predictor_variables,
                  fields_to_scale, scaler):
    # calculate linear curve
    x1 = Xy_predict[predictor_variables[0]].values
    x2 = Xy_predict[predictor_variables[1]].values
    Xy_predict[response_variable] = alpha + beta * x1 + gamma * x2

    # scale back
    if scaler != None:
        Xy_predict[fields_to_scale] = pd.DataFrame(scaler.inverse_transform(Xy_predict[fields_to_scale]),
                                                   columns=Xy_predict[fields_to_scale].columns)

    # scale back from log if necessry
    if response_variable.split("_")[0] == "LOG":
        y_prediction = np.exp(Xy_predict[response_variable].values)
    else:
        y_prediction = Xy_predict[response_variable].values

    return y_prediction


def main(output_trace_path, X_path, main_cities, output_path):
    # loading data
    with open(output_trace_path, 'rb') as buff:
        data = pickle.load(buff)
        hierarchical_model, hierarchical_trace, scaler, degree_index, \
        response_variable, predictor_variables = data['inference'], data['trace'], data['scaler'], \
                                                 data['city_index_df'], data['response_variable'], \
                                                 data['predictor_variables']

    # fields to scale, get data of traces
    fields_to_scale = [response_variable] + predictor_variables

    # get variables
    data = pm.trace_to_dataframe(hierarchical_trace)
    data = data.sample(n=1000).reset_index(drop=True)

    # list of cities
    for city in main_cities:
        path_to_city_data = os.path.join(X_path, city + ".csv")
        X_predict = input_data(path_to_city_data, response_variable, fields_to_scale, scaler)

        alphas = []
        betas = []
        gammas = []
        num_buildings = X_predict.shape[0]
        for building in range(num_buildings):
            building_class = X_predict.loc[building, "BUILDING_CLASS"]
            buildig_city = X_predict.loc[building, "CITY"]
            index = degree_index[(degree_index["CITY"] == buildig_city) & (
                        degree_index["BUILDING_CLASS"] == building_class)].index.values[0]
            alphas.append(data['degree_state_county_b__' + str(index)].mean())
            betas.append(data['degree_state_county_m__' + str(index)].mean())
            gammas.append(data['degree_state_county_g__' + str(index)].mean())

        if response_variable.split("_")[0] == "LOG":
            new_name = response_variable.split("_", 1)[1]
        else:
            new_name = response_variable
        # get input data and scale if necessary
        # do prediction
        X_predict[new_name] = do_prediction(X_predict, alphas, betas,
                                            gammas,
                                            response_variable,
                                            predictor_variables,
                                            fields_to_scale,
                                            scaler)
        # save results per city
        X_predict['EUI_kWh_m2yr'] = a
        X_predict.to_csv(os.path.join(output_path, city + ".csv"))


if __name__ == "__main__":
    name_model = "log_log_all_2var_standard_2500"
    output_path = os.path.join(HIERARCHICAL_MODEL_PREDICTION_FOLDER_2_LEVELS_2_COVARIATE, name_model)
    output_trace_path = os.path.join(HIERARCHICAL_MODEL_INFERENCE_FOLDER_2_LEVELS_2_COVARIATE, name_model + ".pkl")
    X_path = DATA_PREDICTION_FOLDER_TODAY_EFFICIENCY
    main_cities = pd.read_excel(CONFIG_FILE, sheet_name='cities_with_energy_data')['City'].values
    main(output_trace_path, X_path, main_cities, output_path)
