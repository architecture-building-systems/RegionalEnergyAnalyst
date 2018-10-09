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
from models.neural_network_2_variables.definition_and_inference import build_estimator, input_fn, test_fn

def calc_MAPE(y_true, y_pred, n):
    delta = (y_pred - y_true)
    error = np.sum((np.abs(delta/y_true)))*100/n
    return error

def estimate_accurracy(FLAGS, train_data_path, test_data_path, main_cities, output_path, model_name):

    # load neural network
    model = build_estimator(FLAGS['model_dir'], FLAGS['model_type'], FLAGS['cities'], FLAGS['building_classes'], FLAGS['hidden_units'])

    # DO CALCULATION FOR ALL CLASSES IN THE MODEL (CITIES)
    # Upload data to memory and apply scaler of training data to the other variables
    X_train, y_train, _, _ = input_fn(train_data_path, FLAGS, FLAGS['scaler_X'], FLAGS['scaler_y'], train = False)
    X_test, y_test, _, _ = input_fn(test_data_path, FLAGS, FLAGS['scaler_X'], FLAGS['scaler_y'], train = False)

    MAPE_single_building_train,  MAPE_all_buildings_train, R2_train = calc_accurracy(model, FLAGS['scaler_y'], X_train, y_train, FLAGS, model_name)

    # calc accurracy against testing set
    MAPE_single_building_test, MAPE_all_buildings_test,  R2_test = calc_accurracy(model, FLAGS['scaler_y'], X_test, y_test, FLAGS, model_name)

    accurracy_df = pd.DataFrame.from_items([("CITY", ["All", ""]),
                                            ("DATASET", ["Training", "Testing"]),
                                            ("MAPE_building [%]", [MAPE_single_building_train, MAPE_single_building_test]),
                                            ("MAPE_city [%]", [MAPE_all_buildings_train, MAPE_all_buildings_test]),
                                            ("R2 [-]", [R2_train, R2_test])])
    accurracy_df_2 = pd.DataFrame()

    # DO CALCULATION FOR EVERY CLASS IN THE MODEL (CITIES)
    for city in FLAGS['cities']:

        FLAGS['cities'] = [city]
        X_train, y_train, _, _ = input_fn(train_data_path, FLAGS, FLAGS['scaler_X'], FLAGS['scaler_y'], train=False)
        X_test, y_test, _, _ = input_fn(test_data_path, FLAGS, FLAGS['scaler_X'], FLAGS['scaler_y'], train=False)

        MAPE_single_building_train, MAPE_all_buildings_train, R2_train = calc_accurracy(model, FLAGS['scaler_y'],
                                                                                        X_train, y_train, FLAGS, model_name)

        # calc accurracy against testing set
        MAPE_single_building_test, MAPE_all_buildings_test, R2_test = calc_accurracy(model, FLAGS['scaler_y'], X_test,
                                                                                     y_test, FLAGS, model_name)

        dict = pd.DataFrame.from_items([("CITY", [city, "",]),
                                        ("DATASET", ["Training", "Testing"]),
                                        ("MAPE_building [%]",
                                         [MAPE_single_building_train, MAPE_single_building_test]),
                                        ("MAPE_city [%]", [MAPE_all_buildings_train, MAPE_all_buildings_test]),
                                        ("R2 [-]", [R2_train, R2_test])])

        #do this to get the cities in order
        if city in main_cities:
            accurracy_df = pd.concat([accurracy_df, dict], ignore_index=True)
        else:
            accurracy_df_2 = pd.concat([accurracy_df_2, dict], ignore_index=True)

    #append both datasets
    accurracy_df = pd.concat([accurracy_df, accurracy_df_2], ignore_index=True)
    accurracy_df.to_csv(output_path, index=False)


def calc_accurracy(model, scaler_y, X_train, y_train, FLAGS, model_name):
    predictions = model.predict(input_fn=lambda: test_fn(X_train, None, FLAGS))
    y_predicted = np.array(list(p['predictions'] for p in predictions))
    y_predicted = scaler_y.inverse_transform(y_predicted.reshape(-1, 1))
    y = scaler_y.inverse_transform(y_train.reshape(-1, 1))

    # if needed to take back log
    if model_name.split("_")[0] == "log":
        y_predicted = np.exp(y_predicted)
        y = np.exp(y)

    #calcculate error
    MAPE_single_building = calc_MAPE(y_true=y, y_pred=y_predicted, n=len(y)).round(2)
    MAPE_city_scale = calc_MAPE(y_true=sum(y), y_pred=sum(y_predicted), n=1).round(2)
    r2 = r2_score(y_true=y, y_pred=y_predicted).round(2)

    return MAPE_single_building, MAPE_city_scale, r2


def main(_):
    from configuration import NN_MODEL_INFERENCE_FOLDER,  DATA_TRAINING_FILE, DATA_TESTING_FILE, CONFIG_FILE, NN_MODEL_PERFORMANCE_FOLDER
    # import inference
    model = "log_nn_wd_4L_2var"
    model_path = os.path.join(NN_MODEL_INFERENCE_FOLDER, model)
    with open(os.path.join(model_path,'flags.pkl'), 'rb') as fp:
        FLAGS = pickle.load(fp)

    # indicate path to databases
    train_data_path = DATA_TRAINING_FILE
    test_data_path = DATA_TESTING_FILE

    main_cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values
    output_path = os.path.join(NN_MODEL_PERFORMANCE_FOLDER, model + ".csv")


    # # build inference
    estimate_accurracy(FLAGS, train_data_path, test_data_path, main_cities, output_path, model)

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
