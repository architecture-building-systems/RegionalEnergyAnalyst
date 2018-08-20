import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
from configuration import DATA_ALLDATA_FILE


data = pd.read_csv(DATA_ALLDATA_FILE)
y = data["LOG_SITE_ENERGY_MWh_yr"].values
X = data[["LOG_HDD_FLOOR_18_5_C_m2","LOG_CDD_FLOOR_18_5_C_m2"]].as_matrix()


X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

