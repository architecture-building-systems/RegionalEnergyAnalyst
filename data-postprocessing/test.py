from configuration import DATA_TRAINING_FILE, DATA_TESTING_FILE
import pandas as pd
from configuration import DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE, CONFIG_FILE


test_cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values

training = pd.read_csv(DATA_TRAINING_FILE)[["CITY", "LOG_SITE_ENERGY_MWh_yr", "LOG_THERMAL_ENERGY_MWh_yr","BUILDING_CLASS"]]
testing = pd.read_csv(DATA_TESTING_FILE)[["CITY", "LOG_SITE_ENERGY_MWh_yr", "LOG_THERMAL_ENERGY_MWh_yr","BUILDING_CLASS"]]
new_training = training[training["CITY"].isin(test_cities)]
new_testing = testing[testing["CITY"].isin(test_cities)]


new_training['BUILDING_CLASS'] = new_training['BUILDING_CLASS'].apply(lambda x: int(1) if x == "Commercial" else int(0))
new_testing['BUILDING_CLASS'] = new_testing['BUILDING_CLASS'].apply(lambda x: int(1) if x == "Commercial" else int(0))

new_training.rename(columns={"LOG_SITE_ENERGY_MWh_yr":"y", "LOG_THERMAL_ENERGY_MWh_yr": "x1", "BUILDING_CLASS":"x2"}, inplace=True)
new_testing.rename(columns={"LOG_SITE_ENERGY_MWh_yr":"y", "LOG_THERMAL_ENERGY_MWh_yr": "x1", "BUILDING_CLASS":"x2"}, inplace=True)

new_training.to_csv(r"C:\Users\JimenoF\Desktop/training_3_cities.csv", sep=";", index=False)
new_testing.to_csv(r"C:\Users\JimenoF\Desktop/testing_3_cities.csv", sep=";", index=False)

