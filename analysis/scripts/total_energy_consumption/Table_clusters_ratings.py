from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import math
import plotly.graph_objs as go
from plotly.offline import plot
from sklearn import mixture

from configuration import DATA_ALLDATA_FILE, DATA_CLUSTERING_TABLE_FOLDER

def calc_string_distribution(x):
    mean = x[0]
    low = x[1]
    high = x[2]
    ranking = str(round(mean,1))+'\n'+" 95% CI ["+str(round(low,1))+","+str(round(high,1))+"]"
    return ranking

random_state = 170
np.random.RandomState(random_state)
df2 = pd.read_csv(DATA_ALLDATA_FILE)
cities = df2.CITY.unique()
building_classes = df2.BUILDING_CLASS.unique()

df2["CLUSTER_LOG_SITE_EUI_kWh_m2yr"] = np.exp(df2["CLUSTER_LOG_SITE_EUI_kWh_m2yr"])
df2["LOG_SITE_EUI_kWh_m2yr"] = np.exp(df2["LOG_SITE_EUI_kWh_m2yr"])
df2 = df2[['CITY', "BUILDING_CLASS", "CLUSTER_LOG_SITE_EUI_kWh_m2yr", 'LOG_SITE_EUI_kWh_m2yr']]

result = df2.groupby(['CITY', "BUILDING_CLASS", "CLUSTER_LOG_SITE_EUI_kWh_m2yr"], as_index=False).agg(['mean','count','std'])

ci95_hi = []
ci95_lo = []
for i in result.index:
    m, c, s = result.loc[i]
    ci95_hi.append(m + 1.96*s/math.sqrt(c))
    ci95_lo.append(m - 1.96*s/math.sqrt(c))

result['ci95_hi'] = ci95_hi
result['ci95_lo'] = ci95_lo
result = result.reset_index()

result['RANKING_VALUE'] = result[['CLUSTER_LOG_SITE_EUI_kWh_m2yr', 'ci95_lo', 'ci95_hi']].apply(lambda x: calc_string_distribution(x), axis =1)
result = result[['CITY', "BUILDING_CLASS", "RANKING_VALUE"]]
shape_n = result.groupby(['CITY', "BUILDING_CLASS"], as_index=False, sort=True).count()
shape_n = shape_n[['CITY', "BUILDING_CLASS", "RANKING_VALUE"]]

result['RANKING_NAME'] = "empty"

names_ranking = ["High", "Medium-High", "Medium", "Low-medium", "Low"]
for i in range(shape_n.shape[0]):
    city = shape_n.loc[i, 'CITY'].values[0]
    bclass = shape_n.loc[i, 'BUILDING_CLASS'].values[0]
    rvalue = shape_n.loc[i, 'RANKING_VALUE'].values[0]
    result.loc[(result['CITY'] == city) & (result['BUILDING_CLASS'] == bclass), 'RANKING_NAME'] = names_ranking[:rvalue]



final_result = pd.pivot_table(result, index =['CITY', 'BUILDING_CLASS'], columns='RANKING_NAME', values=['RANKING_VALUE'], aggfunc=lambda x: ' '.join(x))
final_result.to_csv(os.path.join(DATA_CLUSTERING_TABLE_FOLDER, "cluster.csv"))

x = 1