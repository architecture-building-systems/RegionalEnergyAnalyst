import os
import pickle
os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,force_device=True"
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
import pandas as pd
import pymc3 as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from configuration import HIERARCHICAL_MODEL_PERFORMANCE_FOLDER, HIERARCHICAL_MODEL_INFERENCE_FOLDER, DATA_TRAINING_FILE, DATA_TESTING_FILE, CONFIG_FILE


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
    sum_delta = (delta.sum() / len(target)) * 100
    return round(sum_delta, 2)

def calc_MAPE(y_true, y_pred, n):
    delta = (y_pred - y_true)
    error = np.sum((np.abs(delta/y_true)))*100/n
    return error

def calc_accurracy(Xy_observed, alpha, beta, response_variable, predictor_variables,
                   fields_to_scale, scaler):
    # calculate linear curve
    x1 = Xy_observed[predictor_variables[0]].values

    #y_prediction = alpha + beta * x1 + gamma * x2 + epsilon * x3
    # y_prediction = np.exp(alpha)*(x1**beta)*(x2**gamma)*(x3**epsilon)
    y_prediction = alpha + beta * x1
    y_target = Xy_observed[response_variable].values

    #scale back
    if scaler != None:
        xy_prediction = Xy_observed.copy()
        xy_prediction[response_variable] = y_prediction
        xy_prediction = pd.DataFrame(scaler.inverse_transform(xy_prediction[fields_to_scale]),
                                                 columns=xy_prediction[fields_to_scale].columns)
        #scale back and get in kWh/yr units
        Xy_observed = pd.DataFrame(scaler.inverse_transform(Xy_observed[fields_to_scale]),
                                                 columns=Xy_observed[fields_to_scale].columns)

    #scale back from log if necessry
    if predictor_variables[0].split("_")[0] == "LOG":
        y_prediction = np.exp(xy_prediction[response_variable].values)
    else:
        y_prediction = xy_prediction[response_variable].values
    if response_variable.split("_")[0] =="LOG":
        y_target = np.exp(Xy_observed[response_variable].values)
    else:
        y_target = Xy_observed[response_variable].values

    # plotting
    # plt.scatter(x=y_prediction, y =y_target)
    # plt.show()

    # se = (standard_error(y_true=y_target, y_pred=y_prediction) * 100).round(2)
    # mse = (mean_squared_error(y_true=y_target, y_pred=y_prediction) * 100).round(2)
    # mae = (mean_absolute_error(y_true=y_target, y_pred=y_prediction) * 100).round(2)
    MAPE_single_building = calc_MAPE(y_true=y_target, y_pred=y_prediction, n = len(y_target)).round(2)
    MAPE_city_scale = calc_MAPE(y_true=sum(y_target), y_pred=sum(y_prediction), n = 1).round(2)
    r2 = r2_score(y_true=y_target, y_pred=y_prediction).round(2)

    return MAPE_single_building, MAPE_city_scale, r2#, se

def main(output_trace_path, Xy_training_path, Xy_testing_path, output_path, main_cities):
    # loading data
    with open(output_trace_path, 'rb') as buff:
        data = pickle.load(buff)
        hierarchical_model, hierarchical_trace, scaler, degree_index, \
        response_variable, predictor_variables = data['inference'], data['trace'], data['scaler'], \
                                                 data['city_index_df'], data['response_variable'],\
                                                 data['predictor_variables']

    # calculate Convergence stats
    bfmi = pm.bfmi(hierarchical_trace).round(2)
    max_gr = max(np.max(gr_stats) for gr_stats in pm.gelman_rubin(hierarchical_trace).values()).round(2)
    n = pm.diagnostics.effective_n(hierarchical_trace)
    efffective_samples_city_beta = n['b1']
    efffective_samples_global_beta = n['global_b1']

    # fields to scale
    fields_to_scale = [response_variable] + predictor_variables

    # get training data scaler
    Xy_training = pd.read_csv(Xy_training_path)
    Xy_testing = pd.read_csv(Xy_testing_path)

    if scaler != None:
        Xy_training[fields_to_scale] = pd.DataFrame(scaler.transform(Xy_training[fields_to_scale]),
                                                columns=Xy_training[fields_to_scale].columns)

        Xy_testing[fields_to_scale] = pd.DataFrame(scaler.transform(Xy_testing[fields_to_scale]),
                                               columns=Xy_testing[fields_to_scale].columns)

    # get data of traces
    data = pm.trace_to_dataframe(hierarchical_trace)

    # DO CALCULATION FOR ALL CLASSES IN THE MODEL (CITIES)
    # get mean coefficeints
    alpha = data['global_b1'].mean()
    beta = data['global_b2'].mean()
    # epsilon = data['global_d'].mean()
    # err = data['eps'].mean()

    # calc accurracy against training set
    # get scaled values for the city
    MAPE_single_building_train,  MAPE_all_buildings_train, R2_train = calc_accurracy(Xy_training, alpha, beta, response_variable,
                                                    predictor_variables, fields_to_scale, scaler)

    # calc accurracy against testing set
    MAPE_single_building_test, MAPE_all_buildings_test,  R2_test = calc_accurracy(Xy_testing, alpha, beta,  response_variable,
                                                 predictor_variables, fields_to_scale, scaler)

    accurracy_df = pd.DataFrame.from_items([("CITY", ["All", ""]),
                                            ("DATASET", ["Training", "Testing"]),
                                            ("MAPE_building [%]", [MAPE_single_building_train, MAPE_single_building_test]),
                                            ("MAPE_city [%]", [MAPE_all_buildings_train, MAPE_all_buildings_test]),
                                            ("R2 [-]", [R2_train, R2_test]),
                                            ("BFMI [-]", [bfmi, ""]),
                                            ("GB [-]", [max_gr, ""]),
                                            ("N_eff",[efffective_samples_global_beta, ""])])
    accurracy_df_2 = pd.DataFrame()

    # DO CALCULATION FOR EVERY CLASS IN THE MODEL (CITIES)
    for i, city in zip(degree_index["CODE"].values, degree_index["CITY"].values):
        # get mean coefficeints
        alpha = data['b1__' + str(i)].mean()
        beta = data['b2__' + str(i)].mean()

        # calc accurracy against training set
        Xy_training_city = Xy_training[Xy_training["CITY"] == city]
        MAPE_single_building_train, MAPE_all_buildings_train, R2_train = calc_accurracy(Xy_training_city, alpha, beta,
                                                        response_variable, predictor_variables, fields_to_scale, scaler)

        # calc accurracy against testing set
        Xy_testing_city = Xy_testing[Xy_testing["CITY"] == city]
        MAPE_single_building_test, MAPE_all_buildings_test, R2_test = calc_accurracy(Xy_testing_city, alpha, beta, response_variable,
                                                     predictor_variables, fields_to_scale, scaler)

        dict = pd.DataFrame.from_items([("CITY", [city, "",]),
                                        ("DATASET", ["Training", "Testing"]),
                                        ("MAPE_building [%]", [MAPE_single_building_train, MAPE_single_building_test]),
                                        ("MAPE_city [%]", [MAPE_all_buildings_train, MAPE_all_buildings_test]),
                                        ("R2 [-]", [R2_train, R2_test]),
                                        ("BFMI [-]", [bfmi, ""]),
                                        ("GB [-]", [max_gr, ""]),
                                        ("N_eff", [efffective_samples_city_beta[i], ""])])

        #do this to get the cities in order
        if city in main_cities:
            accurracy_df = pd.concat([accurracy_df, dict], ignore_index=True)
        else:
            accurracy_df_2 = pd.concat([accurracy_df_2, dict], ignore_index=True)

    #append both datasets
    accurracy_df = pd.concat([accurracy_df, accurracy_df_2], ignore_index=True)
    accurracy_df.to_csv(output_path, index=False)



if __name__ == "__main__":

    name_model = "log_log_all_2var_standard_5000"
    output_path = os.path.join(HIERARCHICAL_MODEL_PERFORMANCE_FOLDER, name_model + ".csv")
    output_trace_path = os.path.join(HIERARCHICAL_MODEL_INFERENCE_FOLDER, name_model + ".pkl")
    Xy_training_path = DATA_TRAINING_FILE
    Xy_testing_path = DATA_TESTING_FILE
    main_cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values

    main(output_trace_path, Xy_training_path, Xy_testing_path, output_path, main_cities)
