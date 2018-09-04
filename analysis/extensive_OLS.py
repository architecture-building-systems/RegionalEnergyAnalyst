from sklearn import linear_model
import pandas as pd
from configuration import CONFIG_FILE, DATA_RAW_BUILDING_PERFORMANCE_FOLDER, DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER, DATA_OPTIMAL_TEMPERATURE_FILE, DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from data_processing.training_and_testing_database import prepare_input_database
from data_raw.IPCC_scenarios.enthalpy_calculation import calc_enthalpy


data_energy_folder = DATA_RAW_BUILDING_PERFORMANCE_FOLDER
data_ipcc_folder = DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER
data_ipcc_file = DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE
scenario = "1990_2010"
cities = pd.read_excel(CONFIG_FILE, sheet_name='cities_with_energy_data')['City'].values
temperatures = [18.5]
final_df = pd.DataFrame()
for temperature in temperatures:
    data_ipcc_file, _ =  calc_enthalpy(cities, data_ipcc_folder, ["data_1990_2010"], temperature)
    dataframe = prepare_input_database(cities, data_energy_folder, data_ipcc_file, scenario)[["CITY","LOG_SITE_ENERGY_MWh_yr", "LOG_THERMAL_ENERGY_MWh_yr", "HKKD_kJ_kg", "SHR"]]
    for city in cities:
        data = dataframe[dataframe["CITY"]==city]
        X = data["LOG_THERMAL_ENERGY_MWh_yr"].values
        y = data["LOG_SITE_ENERGY_MWh_yr"].values
        X = X.reshape(-1,1)

        #fit the model
        reg = linear_model.LinearRegression()
        reg.fit (X, y)

        #do predictions
        y_pred = reg.predict(X)
        fields = {"CITY":[city],
                  "T_C":[temperature],
                  "enthalpy":[data["HKKD_kJ_kg"].values[0]],
                  "b0":[reg.intercept_],
                  "b1":reg.coef_,
                  "R2": [r2_score(y, y_pred)],
                  "MSE":[mean_squared_error(y, y_pred)],
                  "MAE": [mean_absolute_error(y, y_pred)],
                  }
        final_df = pd.concat([final_df, pd.DataFrame(fields)], ignore_index=True)
        x=1
final_df.to_csv(DATA_OPTIMAL_TEMPERATURE_FILE)


