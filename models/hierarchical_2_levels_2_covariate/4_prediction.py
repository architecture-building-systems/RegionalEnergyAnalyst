import os
import pickle

os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,force_device=True"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import pandas as pd
import pymc3 as pm
import math
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
        Xy_predict = pd.DataFrame(scaler.inverse_transform(Xy_predict[fields_to_scale]),
                                                   columns=Xy_predict[fields_to_scale].columns)

    # scale back from log if necessry
    if response_variable.split("_")[0] == "LOG":
        y_prediction = np.exp(Xy_predict[response_variable].values.astype(float))
    else:
        y_prediction = Xy_predict[response_variable]

    return y_prediction


def calc_a_b_g_building(building_class,
                        alpha_commercial,
                        beta_commercial,
                        gamma_commercial,
                        alpha_residential,
                        beta_residential,
                        gamma_residential):
    if building_class == "Residential":
        return alpha_residential, beta_residential, gamma_residential
    elif building_class == "Commercial":
        return alpha_commercial, beta_commercial, gamma_commercial


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
    data = data.sample(n=100).reset_index(drop=True)

    if response_variable.split("_")[0] == "LOG":
        new_name = response_variable.split("_", 1)[1]
    else:
        new_name = response_variable

    # list of cities
    x_prediction_final = pd.DataFrame()
    alphas = []
    betas = []
    gammas = []
    for city in ['Los Angeles, CA']:
        path_to_city_data = os.path.join(X_path, city + ".csv")
        X_prediction = input_data(path_to_city_data, response_variable, fields_to_scale, scaler)
        sceanarios = X_prediction.SCENARIO.unique()
        for sector in ["Residential", "Commercial"]:
            X_predict_sector2 = X_prediction[X_prediction["BUILDING_CLASS"] == sector]
            if X_predict_sector2.empty or X_predict_sector2.empty:
                print(city, sector, "does not exist, we are skipping it")
            else:
                index_sector = degree_index[(degree_index["CITY"] == city) & (degree_index["BUILDING_CLASS"] == sector)].index.values[0]

                for sceanrio in sceanarios:
                    X_predict_sector = X_predict_sector2[X_predict_sector2['SCENARIO'] == sceanrio]

                    #get the median building in terms of GFA
                    X_predict_sector.sort_values(by='GROSS_FLOOR_AREA_m2', inplace=True)
                    below_median = pd.DataFrame(X_predict_sector[X_predict_sector['GROSS_FLOOR_AREA_m2'] > X_predict_sector['GROSS_FLOOR_AREA_m2'].median()].iloc[0]).T
                    below_median.append(pd.DataFrame(X_predict_sector[X_predict_sector['GROSS_FLOOR_AREA_m2'] < X_predict_sector['GROSS_FLOOR_AREA_m2'].median()].iloc[-1]).T)
                    x_prediction_final = x_prediction_final.append([below_median]*(100), ignore_index=True)

                    #append to dataframe final
                    alphas.extend(data['degree_state_county_b__' + str(index_sector)].values)
                    betas.extend(data['degree_state_county_m__' + str(index_sector)].values)
                    gammas.extend(data['degree_state_county_g__' + str(index_sector)].values)
        print("city done: ", city)

    # get input data and scale if necessary
    # do prediction
    x_prediction_final[new_name] = do_prediction(x_prediction_final,
                                               alphas,
                                               betas,
                                               gammas,
                                               response_variable,
                                               predictor_variables,
                                               fields_to_scale,
                                               scaler)
    # save results per city
    x_prediction_final['EUI_kWh_m2yr'] = x_prediction_final[new_name] / x_prediction_final['GROSS_FLOOR_AREA_m2']
    x_prediction_final.to_csv(os.path.join(output_path, "predictions_data.csv"))



if __name__ == "__main__":
    name_model = "log_log_all_2var_standard_2500"
    output_path = os.path.join(HIERARCHICAL_MODEL_PREDICTION_FOLDER_2_LEVELS_2_COVARIATE, name_model)
    output_trace_path = os.path.join(HIERARCHICAL_MODEL_INFERENCE_FOLDER_2_LEVELS_2_COVARIATE, name_model + ".pkl")
    X_path = DATA_PREDICTION_FOLDER_TODAY_EFFICIENCY
    main_cities = pd.read_excel(CONFIG_FILE, sheet_name='cities_with_energy_data')['City'].values
    main(output_trace_path, X_path, main_cities, output_path)
