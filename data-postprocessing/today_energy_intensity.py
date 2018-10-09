import pandas as pd
from configuration import DATA_ALLDATA_FILE, CONFIG_FILE, DATA_TODAY_CONSUMPTION_FILE
import numpy as np

all_data = pd.read_csv(DATA_ALLDATA_FILE)
data_cities = pd.read_excel(CONFIG_FILE, sheet_name='cities_with_energy_data')
test_cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values
cities = data_cities['City'].values
latitudes = data_cities['latitude'].values
longitudes = data_cities['longitude'].values
climates = data_cities['climate'].values


list_of_dicts = []
for city, latitude,longitude, climate  in zip(cities, latitudes, longitudes, climates):
    data = all_data[all_data["CITY"]==city]
    average_EUI_kWh_yr = np.mean(data["SITE_EUI_kWh_m2yr"].values).round(2)
    average_Energy_MWh_yr = np.mean(data["SITE_ENERGY_kWh_yr"].values/1000).round(2)
    number_records = len(data["SITE_ENERGY_kWh_yr"].values)
    std_EUI_kWh_yr = np.std(data["SITE_EUI_kWh_m2yr"].values).round(2)
    std_Energy_MWh_yr = np.std(data["SITE_ENERGY_kWh_yr"].values/1000).round(2)
    if city in test_cities:
        label = 125.50
    else:
        label = 225.50
    list_of_dicts.append({"1_city":city,
                          "2_climate_zone": climate.split(" ")[0],
                          "2.1_climate_description": climate.split(" (")[:-1][0].split(" ", 1)[-1:][0],
                          "3_nrecords": number_records,
                          "4_mean_EUI":average_EUI_kWh_yr,
                          "5_std_EUI": std_EUI_kWh_yr,
                          "6_mean_energy": average_Energy_MWh_yr,
                          "7_std_energy": std_Energy_MWh_yr,
                          "8_m_label":label,
                          "9_lat": latitude,
                          "10_lon": longitude,
                          })

pd.DataFrame(list_of_dicts).to_csv(DATA_TODAY_CONSUMPTION_FILE, sep=";", index=False)
