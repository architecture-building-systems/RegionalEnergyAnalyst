from sklearn import linear_model
import pandas as pd
import numpy as np
import statsmodels.api as sm
from configuration import CONFIG_FILE, DATA_RAW_BUILDING_PERFORMANCE_FOLDER, DATA_RAW_BUILDING_IPCC_SCENARIOS_FOLDER, DATA_OPTIMAL_TEMPERATURE_FILE, DATA_ALLDATA_FILE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from data_processing.training_and_testing_database import prepare_input_database


fields_to_scale = ["LOG_THERMAL_ENERGY_MWh_yr", "LOG_SITE_ENERGY_MWh_yr", "SHR"]
dataframe = pd.read_csv(DATA_ALLDATA_FILE)
dataframe = dataframe[dataframe["CITY"]== "New York, NY"]
# dataframe = dataframe[dataframe["BUILDING_CLASS"]== "Commercial"]
#aframe[fields_to_scale] = scaler.fit_transform(dataframe[fields_to_scale])

new_X = np.log((dataframe["GROSS_FLOOR_AREA_m2"] * dataframe["SHR"]).values)

dataframe["B"] = np.log(dataframe["THERMAL_ENERGY_MWh_yr"]/dataframe["GROSS_FLOOR_AREA_m2"])

# X = dataframe[["LOG_THERMAL_ENERGY_MWh_yr", "B"]].values
X = dataframe["LOG_THERMAL_ENERGY_MWh_yr"].values
X = X.reshape(-1,1)
y = dataframe["LOG_SITE_ENERGY_MWh_yr"].values


#fit the model
reg = linear_model.LinearRegression() #linear_model.Lasso(alpha = 0.1)#
reg.fit (X, y)

#do predictions
y_pred = reg.predict(X)

fields = {"b0":[reg.intercept_],
          "b1":reg.coef_,
          "R2": [r2_score(y, y_pred)],
          "MSE":[mean_squared_error(y, y_pred)],
          "MAE": [mean_absolute_error(y, y_pred)],
          }
print(fields)

import matplotlib.pyplot as plt

plt.plot(y,y_pred, 'ro')
plt.ylabel('some numbers')
plt.show()


X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())



