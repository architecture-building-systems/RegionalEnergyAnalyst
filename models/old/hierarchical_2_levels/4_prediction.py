import os
import pickle

os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,force_device=True"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import numpy as np
import pandas as pd
import pymc3 as pm
from configuration import HIERARCHICAL_MODEL_INFERENCE_FOLDER, DATA_PREDICTION_FOLDER, HIERARCHICAL_MODEL_PREDICTION_FOLDER, CONFIG_FILE


def main(output_trace_path, X_path, main_cities, output_path):
    # loading data
    with open(output_trace_path, 'rb') as buff:
        data = pickle.load(buff)
        hierarchical_model, hierarchical_trace, scaler, degree_index, \
        response_variable, predictor_variables = data['inference'], data['trace'], data['scaler'], \
                                                 data['city_index_df'], data['response_variable'],\
                                                 data['predictor_variables']

    #list of cities
    degree_index.set_index("CITY", inplace=True)

    # get data of traces
    data = pm.trace_to_dataframe(hierarchical_trace)

    # DO CALCULATION FOR EVERY CLASS IN THE MODEL (CITIES)
    for city in main_cities:
        # get input data and scale if necessary
        X = pd.read_csv(os.path.join(X_path, city+".csv"))
        prediction, response_variable_real = calc_prediction(X, city, data, degree_index, predictor_variables, response_variable, scaler)

        # create groups
        df = prediction.groupby("IPCC_SCENARIO")[[response_variable_real]].sum()

        # save results per city
        df.to_csv(os.path.join(output_path, city +".csv"), index_label="IPCC_SCENARIO")
        # save results to disk per building
        # prediction.to_csv(os.path.join(output_path, city + "_building.csv"))

def calc_prediction(X, city, data, degree_index, predictor_variables, response_variable, scaler):

    fields_to_scale = [response_variable] + predictor_variables

    # scale variables
    if scaler != None:
        X[response_variable] = 0.0  # fill in so scaler does not get confused
        X[fields_to_scale] = pd.DataFrame(scaler.transform(X[fields_to_scale]), columns=X[fields_to_scale].columns)

    # get mean regression coefficients
    i = degree_index.loc[city, "CODE"]
    alpha = data['alpha__' + str(i)].mean()
    beta = data['beta__' + str(i)].mean()
    gamma = data['gamma__' + str(i)].mean()

    # get predictors
    x1 = X[predictor_variables[0]].values
    x2 = X[predictor_variables[1]].values

    # predict
    X[response_variable] = alpha + beta * x1 + gamma * x2

    # get back to original units
    if scaler != None:
        X[fields_to_scale] = pd.DataFrame(scaler.inverse_transform(X[fields_to_scale]), columns=X[fields_to_scale].columns)

    #scale back from log if necessry
    if response_variable.split("_")[0] == "LOG":
        X[response_variable] = np.exp(X[response_variable].values)
        new_name = response_variable.split("_",1)[1]
    else:
        new_name = response_variable

    return X.rename(columns={response_variable: new_name}), new_name


if __name__ == "__main__":
    name_model = "log_log_all_standard_10000"
    output_path = os.path.join(HIERARCHICAL_MODEL_PREDICTION_FOLDER, name_model)
    output_trace_path = os.path.join(HIERARCHICAL_MODEL_INFERENCE_FOLDER, name_model + ".pkl")
    X_path = DATA_PREDICTION_FOLDER
    main_cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values
    main(output_trace_path, X_path, main_cities, output_path)
