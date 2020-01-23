import os
import pickle

os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,force_device=True"
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
import pandas as pd
import pymc3 as pm
from sklearn.metrics import r2_score
from configuration import HIERARCHICAL_MODEL_PERFORMANCE_FOLDER_2_LEVELS, HIERARCHICAL_MODEL_INFERENCE_FOLDER_2_LEVELS, \
    DATA_TRAINING_FILE, DATA_TESTING_FILE, CONFIG_FILE


def calc_cv_rmse(prediction, target):
    """
    This function calculates the covariance of the root square mean error between two vectors.
    :param prediction: vector of predicted/simulated data
    :param target: vector of target/measured data
    :return:
        CVrmse: float
        rmse: float
    """

    delta = (prediction - target) ** 2
    sum_delta = delta.sum()
    if sum_delta > 0:
        mean = target.mean()
        n = len(prediction)
        rmse = np.sqrt((sum_delta / n))
        CVrmse = rmse / mean
    else:
        rmse = 0
        CVrmse = 0
    return round(CVrmse * 100, 2), round(rmse, 3)  # keep only 3 significant digits


def calc_mae(prediction, target):
    delta = abs((prediction - target))
    sum_delta = (delta.sum() / len(target)) * 100
    return round(sum_delta, 2)


def calc_mse(prediction, target):
    delta = (prediction - target) ** 2
    sum_delta = (delta.sum() / len(target))
    return round(sum_delta, 2)


def calc_MAPE(y_true, y_pred, n):
    delta = (y_pred - y_true)
    error = np.sum((np.abs(delta / y_true))) * 100 / n
    return error


def do_prediction(Xy_observed, alpha, beta, response_variable, predictor_variables,
                  fields_to_scale, scaler):
    # calculate linear curve
    x1 = Xy_observed[predictor_variables[0]].values
    y_prediction_log_strd = alpha + beta * x1
    y_target_log_strd = Xy_observed[response_variable].values

    # scale back
    if scaler != None:
        xy_prediction = Xy_observed.copy()
        xy_prediction[response_variable] = y_prediction_log_strd
        xy_prediction = pd.DataFrame(scaler.inverse_transform(xy_prediction[fields_to_scale]),
                                     columns=xy_prediction[fields_to_scale].columns)
        # scale back and get in kWh/yr units
        Xy_observed = pd.DataFrame(scaler.inverse_transform(Xy_observed[fields_to_scale]),
                                   columns=Xy_observed[fields_to_scale].columns)

    # scale back from log if necessry
    if predictor_variables[0].split("_")[0] == "LOG":
        y_prediction = np.exp(xy_prediction[response_variable].values)
    else:
        y_prediction = xy_prediction[response_variable].values

    if response_variable.split("_")[0] == "LOG":
        y_target = np.exp(Xy_observed[response_variable].values)
    else:
        y_target = Xy_observed[response_variable].values

    return y_prediction, y_target, y_prediction_log_strd, y_target_log_strd


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
    num_buildings_train = Xy_training.shape[0]
    for building in range(num_buildings_train):
        buildig_city = Xy_training.loc[building, "CITY"]
        index = degree_index[(degree_index["CITY"] == buildig_city)].index_ds.values[0]
        alpha_training.append(data['degree_state_b__' + str(index)].mean())
        beta_training.append(data['degree_state_m__' + str(index)].mean())

    alpha_testing = []
    beta_testing = []
    num_buildings_test = Xy_testing.shape[0]
    for building in range(num_buildings_test):
        buildig_city = Xy_training.loc[building, "CITY"]
        index = degree_index[(degree_index["CITY"] == buildig_city)].index_ds.values[0]
        alpha_testing.append(data['degree_state_b__' + str(index)].mean())
        beta_testing.append(data['degree_state_m__' + str(index)].mean())

    # do for the training data set
    Xy_training["prediction"], Xy_training["observed"], _, _ = do_prediction(Xy_training, alpha_training, beta_training,
                                                                             response_variable,
                                                                             predictor_variables,
                                                                             fields_to_scale,
                                                                             scaler)
    # do for the testing data set
    Xy_testing["prediction"], Xy_testing["observed"], _, _ = do_prediction(Xy_testing, alpha_testing, beta_testing,
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
                                            ("MAPE_build_EUI_%",
                                             [MAPE_single_building_train, MAPE_single_building_test]),
                                            ("MAPE_%", [MAPE_city_scale_train, MAPE_city_scale_test]),
                                            ("n_samples", [n_samples_train, n_samples_test])])


            accurracy_df = pd.concat([accurracy_df, dictionary], ignore_index=True)
    # append both datasets
    accurracy_df.to_csv(output_path, index=False)

def calc_accurracy(y_prediction, y_target):
    MAPE_single_building = calc_MAPE(y_true=y_target, y_pred=y_prediction, n=len(y_target)).round(2)
    MAPE_city_scale = calc_MAPE(y_true=np.mean(y_target), y_pred=np.mean(y_prediction), n=1).round(2)
    r2 = r2_score(y_true=y_target, y_pred=y_prediction).round(2)

    return MAPE_single_building, MAPE_city_scale, r2


def input_data(Xy_testing_path, Xy_training_path, fields_to_scale, scaler):
    # READ DATA
    Xy_training = pd.read_csv(Xy_training_path)
    Xy_testing = pd.read_csv(Xy_testing_path)

    if scaler != None:
        Xy_training[fields_to_scale] = pd.DataFrame(scaler.transform(Xy_training[fields_to_scale]),
                                                    columns=Xy_training[fields_to_scale].columns)

        Xy_testing[fields_to_scale] = pd.DataFrame(scaler.transform(Xy_testing[fields_to_scale]),
                                                   columns=Xy_testing[fields_to_scale].columns)

    return Xy_testing, Xy_training


if __name__ == "__main__":
    name_model = "log_log_all_2var_repram_better_tree_standard_10000"
    output_path = os.path.join(HIERARCHICAL_MODEL_PERFORMANCE_FOLDER_2_LEVELS, name_model + "CITY.csv")
    output_trace_path = os.path.join(HIERARCHICAL_MODEL_INFERENCE_FOLDER_2_LEVELS, name_model + ".pkl")
    Xy_training_path = DATA_TRAINING_FILE
    Xy_testing_path = DATA_TESTING_FILE
    main_cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values

    main(output_trace_path, Xy_training_path, Xy_testing_path, output_path, main_cities)
