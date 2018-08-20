import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
import numpy as np
from configuration import CONFIG_FILE,  DATA_ALLDATA_FILE

data_path = DATA_ALLDATA_FILE
cities_path = CONFIG_FILE
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

#
# df = df[df["BUILDING_CLASS"] == "Residential"]
# fields = ["LOG_SITE_EUI_kWh_m2yr","HDD_GFA", "CDD_GFA"]
# df["HDD_GFA"] = df["HDD_18_5_C"]/ df["GROSS_FLOOR_AREA_m2"]
# df["CDD_GFA"] = df["CDD_18_5_C"]/ df["GROSS_FLOOR_AREA_m2"]
# df["HDD_plus_CDD"] = df["LOG_HDD_FLOOR_18_5_C_m2"]+ df["LOG_CDD_FLOOR_18_5_C_m2"]
#
# df["eHDD_FLOOR_18_5_C_m2"] = np.log(df["HDD_FLOOR_18_5_C_m2"].values * df["CDD_FLOOR_18_5_C_m2"].values)
# sns.pairplot(df, x_vars=["LOG_SITE_ENERGY_MWh_yr","LOG_HDD_FLOOR_18_5_C_m2", "LOG_CDD_FLOOR_18_5_C_m2", "HDD_plus_CDD"],
#                  y_vars=["LOG_SITE_ENERGY_MWh_yr","LOG_HDD_FLOOR_18_5_C_m2", "LOG_CDD_FLOOR_18_5_C_m2", "HDD_plus_CDD"],
#                     size=4, aspect=1,
#                          hue="BUILDING_CLASS",
#                          markers="+", plot_kws = dict(s=50, edgecolor="b", linewidth=1), diag_kws = dict(bins=224),
#                          palette="PuBuGn_d")


df = df[df["CITY"] == "New York, NY"] #"Seattle, WA"

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
fig, axes = plt.subplots(1, 3, figsize=(12,6));
plt.subplots_adjust(wspace=0.4, bottom=0.25 , top=0.7);
commercial = (63/255,192/255,194/255)
residential = (126/255,127/255,132/255)
xaxis_1 = r'$\log(y_{i,j})$'
xaxis_2 = r'$\log(x_{1_{i,j}})$'
xaxis_3 = r'$\log(x_{2_{i,j}})$'
yaxis_2 = r'$\log(y_{i,j})$'
yaxis_3 = r'$\log(y_{i,j})$'
# "rgb(255,209,29)","rgb(126,199,143)","rgb(245,131,69)","rgb(240,75,91)"
# df["LOG_SITE_ENERGY_MWh_yr"].plot(ax=axes[0,0], kind='hist', bins=200, color=color); axes[0,0].set_title('(a)');
df[df["BUILDING_CLASS"] == "Residential"]["LOG_SITE_ENERGY_MWh_yr"].plot(ax=axes[0], kind='hist', bins=200, color=residential,); axes[0].set_title('(a)')
df[df["BUILDING_CLASS"] == "Commercial"]["LOG_SITE_ENERGY_MWh_yr"].plot(ax=axes[0], kind='hist', bins=200, color=commercial)


# df.plot.scatter(ax=axes[1,0], x="LOG_HDD_FLOOR_18_5_C_m2", y ="LOG_SITE_ENERGY_MWh_yr", color=color); axes[1,0].set_title('(d)')
df[df["BUILDING_CLASS"] == "Commercial"].plot.scatter(ax=axes[1], x="LOG_HDD_FLOOR_18_5_C_m2", y ="LOG_SITE_ENERGY_MWh_yr", color=commercial); axes[1].set_title('(b)')
df[df["BUILDING_CLASS"] == "Residential"].plot.scatter(ax=axes[1], x="LOG_HDD_FLOOR_18_5_C_m2", y ="LOG_SITE_ENERGY_MWh_yr", color=residential)

df[df["BUILDING_CLASS"] == "Commercial"].plot.scatter(ax=axes[2], x="LOG_CDD_FLOOR_18_5_C_m2", y ="LOG_SITE_ENERGY_MWh_yr", color=commercial); axes[2].set_title('(c)')
df[df["BUILDING_CLASS"] == "Residential"].plot.scatter(ax=axes[2], x="LOG_CDD_FLOOR_18_5_C_m2", y ="LOG_SITE_ENERGY_MWh_yr", color=residential)

axes[0].set_xlabel(xaxis_1, fontsize=14)
axes[0].legend(["Residential", "Commercial"])
axes[1].set_xlabel(xaxis_2, fontsize=14)
axes[2].set_xlabel(xaxis_3, fontsize=14)
axes[1].set_ylabel(yaxis_2, fontsize=14)
axes[2].set_ylabel(yaxis_3, fontsize=14)

# df.plot.scatter(ax=axes[2,0], x="LOG_HDD_FLOOR_18_5_C_m2", y ="LOG_SITE_ENERGY_MWh_yr", color=color); axes[2,0].set_title('(g)')




# df['B'].plot(ax=axes[0,1]); axes[0,1].set_title('B');
# df['C'].plot(ax=axes[1,0]); axes[1,0].set_title('C');
# df['D'].plot(ax=axes[1,1]); axes[1,1].set_title('D');
plt.show()


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
