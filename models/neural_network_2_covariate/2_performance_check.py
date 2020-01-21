import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error

from models.neural_network_2_covariate.definition_and_inference import build_estimator, input_fn, test_fn


def calc_MAPE(y_true, y_pred, n):
    delta = (y_pred - y_true)
    error = np.sum((np.abs(delta / y_true))) * 100 / n
    return error


def estimate_accurracy(FLAGS, train_data_path, test_data_path, main_cities, output_path, model_name):
    # load neural network
    model = build_estimator(FLAGS['model_dir'], FLAGS['model_type'], FLAGS['cities'], FLAGS['building_classes'],
                            FLAGS['hidden_units'])

    # DO CALCULATION FOR ALL CLASSES IN THE MODEL (CITIES)
    # Upload data to memory and apply scaler of training data to the other variables
    X_train, y_train, _, _ = input_fn(train_data_path, FLAGS, FLAGS['scaler_X'], FLAGS['scaler_y'], train=False)
    X_test, y_test, _, _ = input_fn(test_data_path, FLAGS, FLAGS['scaler_X'], FLAGS['scaler_y'], train=False)

    # calculate and asign predictions to dataframe
    data_train = X_train.copy()
    data_train["y_predicted"], data_train["y_measured"], data_train["y_predicted_log"], data_train["y_measured_log"] = do_predictions(model, FLAGS['scaler_y'], X_train, y_train,
                                                                         FLAGS, model_name)

    data_test = X_test.copy()
    data_test["y_predicted"], data_test["y_measured"], data_test["y_predicted_log"], data_test["y_measured_log"] = do_predictions(model, FLAGS['scaler_y'], X_test, y_test,
                                                                       FLAGS, model_name)

    # calculate the energy use intensity (parameter of interest)
    data = pd.read_csv(train_data_path)
    data = data[data['CITY'].isin(FLAGS['cities'])]
    data.reset_index(inplace=True)
    data_train["y_predicted"] = data_train["y_predicted"] / data["GROSS_FLOOR_AREA_m2"]
    data_train["y_measured"] = data_train["y_measured"] / data["GROSS_FLOOR_AREA_m2"]

    # calculate the energy use intensity (parameter of interest)
    data = pd.read_csv(test_data_path)
    data = data[data['CITY'].isin(FLAGS['cities'])]
    data.reset_index(inplace=True)
    data_test["y_predicted"] = data_test["y_predicted"] / data["GROSS_FLOOR_AREA_m2"]
    data_test["y_measured"] = data_test["y_measured"] / data["GROSS_FLOOR_AREA_m2"]

    accurracy_df = pd.DataFrame()
    accurracy_df_2 = pd.DataFrame()
    for city in FLAGS['cities']:
        data_train_city = data_train[data_train["CITY"] == city]
        data_test_city = data_test[data_test["CITY"] == city]

        for building_class in ["Commercial", "Residential"]:
            data_train_city_building_class = data_train_city[data_train_city["BUILDING_CLASS"] == building_class]
            data_test_city_building_class = data_test_city[data_test_city["BUILDING_CLASS"] == building_class]

            if data_train_city_building_class.empty or data_test_city_building_class.empty:
                print(city, building_class, "does not exist, we are skipping it")
            else:
                # calc accurracy against testing set
                MAPE_single_building_train, MAPE_all_buildings_train, R2_train = calc_accurracy(
                    data_train_city_building_class['y_predicted'].values,
                    data_train_city_building_class['y_measured'].values)

                MAPE_single_building_test, MAPE_all_buildings_test, R2_test = calc_accurracy(
                    data_test_city_building_class['y_predicted'].values,
                    data_test_city_building_class['y_measured'].values)

                MSE_log_domain_train = mean_squared_error(data_train_city_building_class["y_measured_log"].values, data_train_city_building_class["y_predicted_log"].values)
                MSE_log_domain_test =  mean_squared_error(data_test_city_building_class["y_measured_log"].values, data_test_city_building_class["y_predicted_log"].values)

                n_samples_train = data_train_city_building_class['y_measured'].count()
                n_samples_test = data_test_city_building_class['y_measured'].count()

                dict = pd.DataFrame.from_items([("CITY", [city, city]),
                                                ("BUILDING_CLASS", [building_class, building_class]),
                                                ("DATASET", ["Training", "Testing"]),
                                                ("MAPE_build_EUI_%",[MAPE_single_building_train, MAPE_single_building_test]),
                                                ("PE_mean_EUI_%", [MAPE_all_buildings_train, MAPE_all_buildings_test]),
                                                ("MSE_log_domain", [MSE_log_domain_train, MSE_log_domain_test]),
                                                ("n_samples", [n_samples_train, n_samples_test])])

                # do this to get the cities in order
                if city in main_cities:
                    accurracy_df = pd.concat([accurracy_df, dict], ignore_index=True)
                else:
                    accurracy_df_2 = pd.concat([accurracy_df_2, dict], ignore_index=True)

    # append both datasets
    accurracy_df = pd.concat([accurracy_df, accurracy_df_2], ignore_index=True)
    accurracy_df.to_csv(output_path, index=False)


def calc_accurracy(y_predicted, y):
    MAPE_single_building = calc_MAPE(y_true=y, y_pred=y_predicted, n=len(y)).round(2)
    MAPE_city_scale = calc_MAPE(y_true=np.mean(y), y_pred=np.mean(y_predicted), n=1).round(2)
    r2 = r2_score(y_true=y, y_pred=y_predicted).round(2)

    return MAPE_single_building, MAPE_city_scale, r2


def do_predictions(model, scaler_y, X_train, y_train, FLAGS, model_name):
    predictions = model.predict(input_fn=lambda: test_fn(X_train, None, FLAGS))
    y_predicted_log = np.array(list(p['predictions'] for p in predictions))
    y_measured_log = y_train

    y_predicted_log_no_std = scaler_y.inverse_transform(y_predicted_log.reshape(-1, 1))
    y_measured_log_no_std = scaler_y.inverse_transform(y_train.reshape(-1, 1))

    # if needed to take back log
    if model_name.split("_")[0] == "log":
        y_predicted = np.exp(y_predicted_log_no_std)
        y_measured = np.exp(y_measured_log_no_std)

    return y_predicted, y_measured, y_predicted_log, y_measured_log,


def main(_):
    from configuration import NN_MODEL_INFERENCE_FOLDER, DATA_TRAINING_FILE, DATA_TESTING_FILE, CONFIG_FILE, \
        NN_MODEL_PERFORMANCE_FOLDER
    # import inference
    model = "log_nn_wd_4L_2var"
    model_path = os.path.join(NN_MODEL_INFERENCE_FOLDER, model)
    with open(os.path.join(model_path, 'flags.pkl'), 'rb') as fp:
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
