import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
import numpy as np

data_path = os.path.join(os.path.dirname(os.getcwd()), "data_processing", "data", "input_database.csv")
cities_path = os.path.join(os.path.dirname(os.getcwd()), "cities.xlsx")
cities = pd.read_excel(cities_path, sheet_name='test_cities')['City'].values
# fields = ["SITE_ENERGY_kWh/yr", "SITE_EUI_kWh_m2yr", "GROSS_FLOOR_AREA_m2", "HDD_18.5_C", "CDD_18.5_C", "LOG_SITE_EUI_kWh/m2.yr"]

df = pd.read_csv(data_path)
# df = df[df.CITY.isin(cities)]   # "New York, NY"]


# df["ok"] = df["LOG_HDD_FLOOR_18_5_C_m2"] * df["LOG_CDD_FLOOR_18_5_C_m2"]
fields = ["LOG_SITE_ENERGY_MWh_yr", "LOG_HDD_FLOOR_18_5_C_m2", "LOG_CDD_FLOOR_18_5_C_m2"]  # , "ok"]

# df["HDD_18_5_C"] = df["HDD_18_5_C"]*24/1000000
# df["CDD_18_5_C"] = df["CDD_18_5_C"]*24/1000000
# df["LOG_HDD_FLOOR_18_5_C_m2"] = np.log(df["HDD_18_5_C"]*df["GROSS_FLOOR_AREA_m2"])
# df["LOG_CDD_FLOOR_18_5_C_m2"] = np.log(df["CDD_18_5_C"]*df["GROSS_FLOOR_AREA_m2"])

#scatter_matrix(df[fields], alpha=0.2, marker='o', figsize=(10, 10), diagonal='hist', hist_kwds={'bins': 224})
# df = df[df["BUILDING_CLASS"] == "Commercial"]
# sns.pairplot(df, x_vars=["LOG_SITE_ENERGY_MWh_yr", "LOG_HDD_FLOOR_18_5_C_m2",
#                          "LOG_CDD_FLOOR_18_5_C_m2"], y_vars=["LOG_SITE_ENERGY_MWh_yr", "LOG_HDD_FLOOR_18_5_C_m2",
#                          "LOG_CDD_FLOOR_18_5_C_m2"], size=4, aspect=1,
#                          hue="BUILDING_CLASS",
#                          #diag_kind='none',
#                          markers="+", plot_kws = dict(s=50, edgecolor="b", linewidth=1), diag_kws = dict(bins=224),
#                          palette="PuBuGn_d")



fields = ["LOG_SITE_EUI_kWh_m2yr","HDD_GFA", "CDD_GFA"]
df["HDD_GFA"] = df["HDD_18_5_C"]/ df["GROSS_FLOOR_AREA_m2"]
df["CDD_GFA"] = df["CDD_18_5_C"]/ df["GROSS_FLOOR_AREA_m2"]
sns.pairplot(df, x_vars=["SITE_EUI_kWh_m2yr", "HDD_18_5_C",
                         "CDD_18_5_C"], y_vars=["SITE_EUI_kWh_m2yr", "HDD_18_5_C",
                         "CDD_18_5_C"], size=4, aspect=1,
                         hue="BUILDING_CLASS",
                         #diag_kind='none',
                         markers="+", plot_kws = dict(s=50, edgecolor="b", linewidth=1), diag_kws = dict(bins=224),
                         palette="PuBuGn_d")

# for city in cities[:2]:
#     city_data = df[df.CITY == city]
#     sns.distplot(a = city_data["LOG_SITE_ENERGY_MWh_yr"])

# sns.violinplot(x="CITY", y="LOG_SITE_ENERGY_MWh_yr", hue="BUILDING_CLASS", data=df,
#                split=True, palette="Set3");

# sns.violinplot(x="LOG_SITE_ENERGY_MWh_yr", hue="BUILDING_CLASS", data=df,
#                split=True, palette="Set3");

# f, ax = plt.subplots(figsize=(10, 8))
# corr = df[fields].corr()
# sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
#             square=True, ax=ax)


# fields = ["LOG_SITE_EUI_kWh_m2yr", "LOG_GROSS_FLOOR_AREA_m2", "HDD_18_5_C", "CDD_18_5_C"]
# df_commercial = df[df.BUILDING_CLASS == "Commercial"]
# scatter_matrix(df_commercial[fields], alpha=0.2, figsize=(10, 10), diagonal='hist', hist_kwds={'bins': 160})
# print(len(df_commercial.index))
# plt.show()
#
# fields = ["LOG_SITE_EUI_kWh_m2yr", "LOG_GROSS_FLOOR_AREA_m2", "HDD_18_5_C", "CDD_18_5_C"]
# df_residential = df[df.BUILDING_CLASS == "Residential"]
# scatter_matrix(df_residential[fields], alpha=0.2, figsize=(10, 10), diagonal='hist', hist_kwds={'bins': 160})
# print(len(df_residential.index))
plt.show()
