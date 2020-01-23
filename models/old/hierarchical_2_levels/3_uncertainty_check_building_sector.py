import os
import pickle

os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,force_device=True"
os.environ["MKL_THREADING_LAYER"] = "GNU"
import numpy as np
import pandas as pd
import pymc3 as pm
from sklearn.metrics import r2_score
from configuration import HIERARCHICAL_MODEL_PERFORMANCE_FOLDER_2_LEVELS, HIERARCHICAL_MODEL_INFERENCE_FOLDER_2_LEVELS, \
    DATA_TRAINING_FILE, DATA_TESTING_FILE, CONFIG_FILE, DATA_PREDICTION_FOLDER_TODAY_EFFICIENCY, DATA_ALLDATA_FILE


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


def do_prediction(Xy_prediction, alpha, beta, response_variable, predictor_variables,
                  fields_to_scale, scaler):
    # calculate linear curve
    x1 = Xy_prediction[predictor_variables[0]].values
    y_prediction_log_strd = alpha + beta * x1

    # scale back
    if scaler != None:
        xy_prediction = Xy_prediction.copy()
        xy_prediction[response_variable] = y_prediction_log_strd
        xy_prediction = pd.DataFrame(scaler.inverse_transform(xy_prediction[fields_to_scale]),
                                     columns=xy_prediction[fields_to_scale].columns)

    # scale back from log if necessry
    if predictor_variables[0].split("_")[0] == "LOG":
        y_prediction = np.exp(xy_prediction[response_variable].values)
    else:
        y_prediction = xy_prediction[response_variable].values

    return y_prediction


def main(output_trace_path, Xy_prediction_path, Xy_testing_path, output_path, main_cities):
    # loading data
    with open(output_trace_path, 'rb') as buff:
        data = pickle.load(buff)
        hierarchical_model, hierarchical_trace, scaler, degree_index, \
        response_variable, predictor_variables = data['inference'], data['trace'], data['scaler'], \
                                                 data['city_index_df'], data['response_variable'], \
                                                 data['predictor_variables']

    #get the city where to do the analysis form
    building_class = "Commercial"
    city = "New York, NY"
    year = 2010
    error = 25
    Xy_prediction_path = os.path.join(Xy_prediction_path, city+".csv")

    # fields to scale, get data of traces
    fields_to_scale = [response_variable] + predictor_variables

    Xy_testing, Xy_prediction = input_data(Xy_testing_path, Xy_prediction_path, fields_to_scale, scaler)

    # get data of traces and only 1000 random samples
    data = pm.trace_to_dataframe(hierarchical_trace)
    data = data.sample(n=1000).reset_index(drop=True)

    # GET only buildings belonging to one calss and city
    index = degree_index[(degree_index["CITY"] == city) & (degree_index["BUILDING_CLASS"] == building_class)].index.values[0]
    alpha_training =data['degree_state_county_b__' + str(index)].tolist()
    beta_training = data['degree_state_county_m__' + str(index)].tolist()

    Xy_prediction = Xy_prediction[Xy_prediction["BUILDING_CLASS"] == building_class]
    Xy_prediction = Xy_prediction[Xy_prediction["CITY"] == city]
    Xy_prediction = Xy_prediction[Xy_prediction["YEAR"] == year]

    Xy_testing = Xy_testing[Xy_testing["BUILDING_CLASS"] == building_class]
    Xy_testing = Xy_testing[Xy_testing["CITY"] == city]
    Xy_testing2 = Xy_testing.sort_values("GROSS_FLOOR_AREA_m2")
    Xy_testing2.reset_index(inplace=True, drop=True)


    x = pd.DataFrame()
    sample_no = 0
    for scenario in ["B1_"+str(year), "A2_"+str(year), "A1B_"+str(year)]:
        Xy_prediction2 = Xy_prediction[Xy_prediction["SCENARIO"] == scenario]
        Xy_prediction2 = Xy_prediction2.sort_values("GROSS_FLOOR_AREA_m2")
        Xy_prediction2.reset_index(inplace=True, drop=True)
        x["GROSS_FLOOR_AREA_m2"] = Xy_prediction2["GROSS_FLOOR_AREA_m2"]

        #this is all the data for today consumption


        for alpha, beta in zip(alpha_training,beta_training):
            #iterate over every sample of alpha and beta and calculate the demand in the training dataset
            Xy_prediction2["prediction"] = do_prediction(Xy_prediction2, alpha,
                                                             beta,
                                                             response_variable,
                                                             predictor_variables,
                                                             fields_to_scale,
                                                             scaler)
            x["sample_"+str(sample_no)] = Xy_prediction2["prediction"] / Xy_prediction2["GROSS_FLOOR_AREA_m2"]
            sample_no+=1

    quantiles_and_median = x.quantile([0.25, 0.5, 0.75], axis=1).T
    x["quantile_1"] = quantiles_and_median[0.25]
    x["median"] = quantiles_and_median[0.50]
    x["quantile_3"] = quantiles_and_median[0.75]

    x["error_low"] = x["median"]*(1+error/100)
    x["error_high"] = x["median"]*(1-error/100)
    x["error_real"] = Xy_testing2["SITE_EUI_kWh_m2yr"]


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(x["GROSS_FLOOR_AREA_m2"].values, x["median"].values, color='black',label='prediction')
    # ax.fill_between(x["GROSS_FLOOR_AREA_m2"].values, x["error_low"].values, x["error_high"].values, color='red')
    ax.fill_between(x["GROSS_FLOOR_AREA_m2"].values, x["quantile_1"].values, x["quantile_3"].values, color='blue',label='credible interval')
    ax.fill_between(x["GROSS_FLOOR_AREA_m2"].values, x["median"].values, x["error_real"].values, color='red',label='distance to observation')
    ax.set_ylim([0, 3500])
    ax.legend(loc='upper right')
    ax.set_ylabel("Energy Use Intensity [kWh/m2.yr]")
    ax.set_xlabel("Gross floor area [m2.yr]")

    plt.show()
    x=1


def calc_accurracy(y_prediction, y_target):
    MAPE_single_building = calc_MAPE(y_true=y_target, y_pred=y_prediction, n=len(y_target)).round(2)
    MAPE_city_scale = calc_MAPE(y_true=np.median(y_target), y_pred=np.median(y_prediction), n=1).round(2)
    r2 = r2_score(y_true=y_target, y_pred=y_prediction).round(2)

    return MAPE_single_building, MAPE_city_scale, r2


def input_data(Xy_testing_path, Xy_prediction_path, fields_to_scale, scaler):
    # READ DATA
    Xy_prediction = pd.read_csv(Xy_prediction_path)
    Xy_prediction['LOG_SITE_ENERGY_kWh_yr'] = 2  # this is just a hack, we do not use it in reality
    Xy_testing = pd.read_csv(Xy_testing_path)

    if scaler != None:
        Xy_prediction[fields_to_scale] = pd.DataFrame(scaler.transform(Xy_prediction[fields_to_scale]),
                                                    columns=Xy_prediction[fields_to_scale].columns)

    return Xy_testing, Xy_prediction


if __name__ == "__main__":
    name_model = "log_log_all_2var_standard_10000"
    output_path = os.path.join(HIERARCHICAL_MODEL_PERFORMANCE_FOLDER_2_LEVELS, name_model + "CLIMATE_ZONE.csv")
    output_trace_path = os.path.join(HIERARCHICAL_MODEL_INFERENCE_FOLDER_2_LEVELS, name_model + ".pkl")
    Xy_prediction_path = DATA_PREDICTION_FOLDER_TODAY_EFFICIENCY
    Xy_testing_path = DATA_ALLDATA_FILE
    main_cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values

    main(output_trace_path, Xy_prediction_path, Xy_testing_path, output_path, main_cities)
