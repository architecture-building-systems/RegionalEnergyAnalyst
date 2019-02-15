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
    y_prediction_log = alpha + beta * x1
    y_target_log = Xy_observed[response_variable].values

    # scale back
    if scaler != None:
        xy_prediction = Xy_observed.copy()
        xy_prediction[response_variable] = y_prediction_log
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

    return y_prediction, y_target, y_prediction_log, y_target_log


def main(output_trace_path, Xy_training_path, Xy_testing_path, output_path, main_cities):
    # loading data
    with open(output_trace_path, 'rb') as buff:
        data = pickle.load(buff)
        hierarchical_model, hierarchical_trace, scaler, degree_index, \
        response_variable, predictor_variables = data['inference'], data['trace'], data['scaler'], \
                                                 data['city_index_df'], data['response_variable'], \
                                                 data['predictor_variables']

    # calculate Convergence stats
    bfmi = pm.bfmi(hierarchical_trace).round(2)
    max_gr = max(np.max(gr_stats) for gr_stats in pm.gelman_rubin(hierarchical_trace).values()).round(2)
    n = pm.diagnostics.effective_n(hierarchical_trace)
    # efffective_samples_city_beta = n['b1']

    # fields to scale, get data of traces
    fields_to_scale = [response_variable] + predictor_variables
    Xy_testing, Xy_training = input_data(Xy_testing_path, Xy_training_path, fields_to_scale, scaler)

    # get data of traces
    data = pm.trace_to_dataframe(hierarchical_trace)
    # GET 1000 RANDOM SAMPLES
    data = data.sample(n=1000).reset_index()

    # import matplotlib.pyplot as plt
    # pm.traceplot(hierarchical_trace, combined=True)
    # plt.show()

    # get data aggreagations:
    train_climate_zones = Xy_training.pivot_table(index=['CLIMATE_ZONE'], values=fields_to_scale, aggfunc=np.mean)
    train_climate_zones_cities = Xy_training.pivot_table(index=["CITY"], values=fields_to_scale,
                                                         aggfunc=np.mean)
    train_climate_zones_cities_building_class = Xy_training.pivot_table(
        index=['CLIMATE_ZONE', "CITY", "BUILDING_CLASS"], values=fields_to_scale, aggfunc=np.mean)

    # get data aggreagations:
    test_climate_zones = Xy_testing.pivot_table(index=['CLIMATE_ZONE'], values=fields_to_scale, aggfunc=np.mean)
    test_climate_zones_cities = Xy_testing.pivot_table(index=["CITY"], values=fields_to_scale,
                                                       aggfunc=np.mean)
    test_climate_zones_cities_building_class = Xy_testing.pivot_table(index=['CLIMATE_ZONE', "CITY", "BUILDING_CLASS"],
                                                                      values=fields_to_scale, aggfunc=np.mean)


    accurracy_df = pd.DataFrame()
    index_city = degree_index.pivot_table(index=['CITY'], values=["index_ds"], aggfunc=np.mean)
    for index, climate in zip(index_city["index_ds"].values, index_city.index.values):
        index = 84
        bclass = 131 # commercial
        climate = "New York, NY"
        # calc accurracy against training set
        Xy_training_city = Xy_training[Xy_training["CITY"] == climate]
        Xy_training_city = Xy_training_city[Xy_training_city["BUILDING_CLASS"] == "Commercial"].reset_index()
        Xy_training_city = Xy_training_city[Xy_training_city.index == 0]
        Xy_testing_city = Xy_testing[Xy_testing["CITY"] == climate]
        Xy_testing_city = Xy_testing_city[Xy_testing_city["BUILDING_CLASS"] == "Commercial"].reset_index()
        Xy_testing_city = Xy_testing_city[Xy_testing_city.index == 0]

        if Xy_training_city.empty or Xy_testing_city.empty:
            print(climate, "does not exist, we are skipping it")
        else:
            y_predictions_train = []
            y_targets_train = []
            y_predictions_test = []
            y_targets_test = []

            for i in range(data.shape[0]):
                alpha = data.loc[i, 'degree_state_county_b__' + str(bclass)]
                beta = data.loc[i, 'degree_state_county_m__' + str(bclass)]

                # do for the training data set
                y_prediction, y_target, y_prediction_log, y_target_log = do_prediction(Xy_training_city, alpha,
                                                                                       beta,
                                                                                       response_variable,
                                                                                       predictor_variables,
                                                                                       fields_to_scale,
                                                                                       scaler)
                y_predictions_train.extend(y_prediction)
                y_targets_train.extend(y_target)

                # do for the testing data set
                y_prediction, y_target, y_prediction_log, y_target_log = do_prediction(Xy_testing_city, alpha, beta,
                                                                                       response_variable,
                                                                                       predictor_variables,
                                                                                       fields_to_scale,
                                                                                       scaler)
                y_predictions_test.extend(y_prediction)
                y_targets_test.extend(y_target)

        MAPE_single_building_train, MAPE_city_scale_train, r2_test = calc_accurracy(np.array(y_predictions_train),
                                                                                    np.array(y_targets_train))
        MAPE_single_building_test, MAPE_city_scale_test, r2_test = calc_accurracy(np.array(y_predictions_test),
                                                                                  np.array(y_targets_test))
        #
        pd.DataFrame(
            {"x1": y_predictions_train, "y1": y_targets_train,
             "MAPE_MEAN_x1": MAPE_city_scale_train,
             "MAPE_ALLPOINTS_x1": MAPE_single_building_train,
             }).to_csv(
            r'C:\Users\JimenoF\Desktop/testst_city.csv')

        pd.DataFrame(
            { "x2": y_predictions_test, "y2": y_targets_test,
             "MAPE_MEAN_x2": MAPE_city_scale_test,
              "MAPE_ALLPOINTS_x2": MAPE_single_building_test,
             }).to_csv(
            r'C:\Users\JimenoF\Desktop/testst_city_test.csv')
        x=1


    # DO CALCULATION FOR AVERAGE BUILDING
    accurracy_df = pd.DataFrame()
    index_climatezone = degree_index.pivot_table(index=['CLIMATE_ZONE'])

    for index, climate in zip(index_climatezone["index_d"].values, index_climatezone.index.values):

        # calc accurracy against training set
        Xy_training_city = train_climate_zones[train_climate_zones.index == climate]
        Xy_testing_city = test_climate_zones[test_climate_zones.index == climate]

        if Xy_training_city.empty or Xy_testing_city.empty:
            print(climate, "does not exist, we are skipping it")
        else:
            y_predictions_train = []
            y_targets_train = []
            y_predictions_test = []
            y_targets_test = []

            for i in range(data.shape[0]):
                alpha = data.loc[i, 'degree_b__' + str(index)]
                beta = data.loc[i, 'degree_m__' + str(index)]

                # do for the training data set
                y_prediction, y_target, y_prediction_log, y_target_log = do_prediction(Xy_training_city, alpha, beta,
                                                                                       response_variable,
                                                                                       predictor_variables,
                                                                                       fields_to_scale,
                                                                                       scaler)
                y_predictions_train.extend(y_prediction)
                y_targets_train.extend(y_target)

                # do for the testing data set
                y_prediction, y_target, y_prediction_log, y_target_log = do_prediction(Xy_testing_city, alpha, beta,
                                                                                       response_variable,
                                                                                       predictor_variables,
                                                                                       fields_to_scale,
                                                                                       scaler)
                y_predictions_test.extend(y_prediction)
                y_targets_test.extend(y_target)

        MAPE_single_building_train, MAPE_city_scale_train, r2_test = calc_accurracy(np.array(y_predictions_train), np.array(y_targets_train))
        MAPE_single_building_test, MAPE_city_scale_test, r2_test = calc_accurracy(np.array(y_predictions_test), np.array(y_targets_test))
        #
        pd.DataFrame(
            {"x1": y_predictions_train, "y1": y_targets_train, "x2": y_predictions_test, "y2": y_targets_test,
             "MAPE_MEAN_x1": MAPE_city_scale_train, "MAPE_MEAN_x2": MAPE_city_scale_test,
             "MAPE_ALLPOINTS_x1": MAPE_single_building_train, "MAPE_ALLPOINTS_x2": MAPE_single_building_test,
             }).to_csv(
            r'C:\Users\JimenoF\Desktop/testst_climate.csv')






        x = 1
        # n_samples_test = len(y_target)
        # MAPE_single_building_train, MAPE_city_scale_train, r2_test = calc_accurracy(y_prediction, y_target)
        # MAPE_single_building_test, MAPE_city_scale_test, r2_test = calc_accurracy(y_prediction, y_target)
        #
        # dict = pd.DataFrame.from_items([("CLIMATE", [climate, climate, ]),
        #                                 ("DATASET", ["Training", "Testing"]),
        #                                 ("MAPE_build_EUI_%",
        #                                  [MAPE_single_building_train, MAPE_single_building_test]),
        #                                 ("PE_mean_EUI_%", [MAPE_city_scale_train, MAPE_city_scale_test]),
        #                                 ("MSE_log_domain", [MSE_log_domain_train, MSE_log_domain_test]),
        #                                 ("n_samples", [n_samples_train, n_samples_test])])
        #
        # accurracy_df = pd.concat([accurracy_df, dict], ignore_index=True)


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
    name_model = "log_log_all_2var_standard_5000"
    output_path = os.path.join(HIERARCHICAL_MODEL_PERFORMANCE_FOLDER_2_LEVELS, name_model + ".csv")
    output_trace_path = os.path.join(HIERARCHICAL_MODEL_INFERENCE_FOLDER_2_LEVELS, name_model + ".pkl")
    Xy_training_path = DATA_TRAINING_FILE
    Xy_testing_path = DATA_TESTING_FILE
    main_cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values

    main(output_trace_path, Xy_training_path, Xy_testing_path, output_path, main_cities)
