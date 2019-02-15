import argparse
import shutil
import sys
import pandas as pd
import os
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pickle
from definition import build_estimator, test_fn

def calc_MAPE(y_true, y_pred, n):
    delta = (y_pred - y_true)
    error = np.sum((np.abs(delta/y_true)))*100/n
    return error

def input_fn(data_file, FLAGS, scaler_X):
    """Generate an input function for the Estimator."""
    data = pd.read_csv(data_file)
    data = data[data['CITY'].isin(FLAGS['cities'])]
    data.reset_index(inplace=True)
    fields_to_scale = FLAGS["fields_to_scale"]
    data[fields_to_scale] = pd.DataFrame(scaler_X.transform(data[fields_to_scale]),columns=data[fields_to_scale].columns)

    features = data[FLAGS['predictor_variables']]
    return data, features


def estimate_accurracy(FLAGS, X_path, main_cities, output_path, model_name):

    # load neural network
    model = build_estimator(FLAGS['model_dir'], FLAGS['model_type'], FLAGS['cities'], FLAGS['building_classes'], FLAGS['hidden_units'])

    # DO CALCULATION FOR EVERY CLASS IN THE MODEL (CITIES)
    for city in main_cities:

        FLAGS['cities'] = [city]
        data, X, = input_fn(os.path.join(X_path, city+".csv"), FLAGS, FLAGS['scaler_X'])

        prediction, response_variable_real = calc_prediction(data, model, X, FLAGS['scaler_y'], FLAGS, model_name)

        # create groups
        df = prediction.groupby("IPCC_SCENARIO")[[response_variable_real]].sum()

        # save results per city
        df.to_csv(os.path.join(output_path, city +".csv"), index_label="IPCC_SCENARIO")

def calc_prediction(data, model, X, scaler_y, FLAGS, model_name):

    predictions = model.predict(input_fn=lambda: test_fn(X, None, FLAGS))
    y_predicted = np.array(list(p['predictions'] for p in predictions))
    y_predicted = scaler_y.inverse_transform(y_predicted.reshape(-1, 1))

    # if needed to take back log
    response_variable_real = "SITE_ENERGY_MWh_yr"
    if model_name.split("_")[0] == "log":
        data[response_variable_real] = np.exp(y_predicted)

    return data, response_variable_real


def main(argv):

    # import inference
    model = "log_neural_net_wide_deep_4L_453%_3%"
    model_path = os.path.join(os.getcwd(), "results", "inference", model)
    with open(os.path.join(model_path,'flags.pkl'), 'rb') as fp:
        FLAGS = pickle.load(fp)

    # indicate path to databases
    X_path = os.path.join(os.getcwd(), "data_processing", "data", "prediction")
    main_cities = pd.read_excel(os.path.join(os.getcwd(), "cities.xlsx"), sheet_name='test_cities')['City'].values
    output_path = os.path.join(os.getcwd(), "results", "predictions", model)

    # # build inference
    estimate_accurracy(FLAGS, X_path,  main_cities, output_path, model)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
