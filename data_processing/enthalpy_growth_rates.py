import os

import pandas as pd
import re

from configuration import DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE, CONFIG_FILE, \
    DATA_RAW_BUILDING_ENTHALPY_FOLDER, DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE_EFFICEINCY


def main(cities, latitudes, longitudes, climates, test_cities, scenarios, flag_use_efficiency):
    for scenario in scenarios:
        if flag_use_efficiency == True:
            all_data = pd.read_csv(DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE_EFFICEINCY)
            output = os.path.join(DATA_RAW_BUILDING_ENTHALPY_FOLDER, "future_enthalpy_days_efficiency_" + scenario + ".csv")
        else:
            all_data = pd.read_csv(DATA_RAW_BUILDING_IPCC_SCENARIOS_ENTHALPY_FILE)
            output = os.path.join(DATA_RAW_BUILDING_ENTHALPY_FOLDER, "future_enthalpy_days_" + scenario + ".csv")
        list_of_dicts = []
        data2 = all_data[all_data["Scenario"] == scenario]
        for city, latitude, longitude, climate in zip(cities, latitudes, longitudes, climates):
            data = data2[data2["City"] == city]
            enthalpy_total = data["HKKD_kJ_kg"].values[0]
            enthalpy_H_sen = data["HKKD_H_sen_kJ_kg"].values[0]
            enthalpy_C_sen = data["HKKD_C_sen_kJ_kg"].values[0]
            enthalpy_HUM = data["HKKD_HUM_kJ_kg"].values[0]
            enthalpy_DEHUM = data["HKKD_DEHUM_kJ_kg"].values[0]
            shr = data["SHR"].values[0]
            if city in test_cities:
                label = 125.50
            else:
                label = 225.50
            climate_zone = climate.split(" ")[0]
            list_of_dicts.append({"1_city": city,
                                  "2_climate_zone": climate_zone,
                                  "20_climate_zone": re.findall('\d+',climate_zone)[0],
                                  "3_climate_description": climate.split(" (")[:-1][0].split(" ", 1)[-1:][0],
                                  "4_enthalpy_H": round(enthalpy_H_sen, 2),
                                  "5_enthalpy_C": round(enthalpy_C_sen, 2),
                                  "6_enthalpy_HUM": round(enthalpy_HUM, 2),
                                  "7_enthalpy_DEHUM": round(enthalpy_DEHUM, 2),
                                  "8_enthalpy_total": round(enthalpy_total, 2),
                                  "9_shr": round(shr, 2),
                                  "10_m_label": label,
                                  "11_lat": latitude,
                                  "12_lon": longitude,
                                  })
        df_output = pd.DataFrame(list_of_dicts)

        if scenario == "1990_2010":
            df_baseline = df_output
        else:
            delta_enthalpy_H = (df_output["4_enthalpy_H"] - df_baseline["4_enthalpy_H"])
            delta_enthalpy_C = (df_output["5_enthalpy_C"] - df_baseline["5_enthalpy_C"])
            delta_enthalpy_HUM = (df_output["6_enthalpy_HUM"] - df_baseline["6_enthalpy_HUM"])
            delta_enthalpy_DEHUM = (df_output["7_enthalpy_DEHUM"] - df_baseline["7_enthalpy_DEHUM"])
            delta_enthalpy_total = df_output["8_enthalpy_total"] - df_baseline["8_enthalpy_total"]

            df_output["13_per_enthalpy_H"] = (delta_enthalpy_H/ df_baseline["4_enthalpy_H"] * 100/9).round(2)
            df_output["14_per_enthalpy_C"] = (delta_enthalpy_C/ df_baseline["5_enthalpy_C"] * 100/9).round(2)
            df_output["15_per_enthalpy_HUM"] = (delta_enthalpy_HUM/ df_baseline["6_enthalpy_HUM"] * 100/9).round(2)
            df_output["16_per_enthalpy_DEHUM"] = (delta_enthalpy_DEHUM/ df_baseline["7_enthalpy_DEHUM"] * 100/9).round(2)
            df_output["17_per_enthalpy_total"] = (delta_enthalpy_total/ df_baseline["8_enthalpy_total"] * 100/9).round(2)

            df_output["13_growth_enthalpy_H"] = (delta_enthalpy_H/9).round(2)
            df_output["14_growth_enthalpy_C"] = (delta_enthalpy_C/9).round(2)
            df_output["15_growth_enthalpy_HUM"] = (delta_enthalpy_HUM/9).round(2)
            df_output["16_growth_enthalpy_DEHUM"] = (delta_enthalpy_DEHUM/9).round(2)
            df_output["17_growth_enthalpy_total"] = (delta_enthalpy_total/9).round(2)

        df_output.to_csv(output, sep=";", index=False)


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.getcwd()), 'IPCC_scenarios', 'data')
    data_cities = pd.read_excel(CONFIG_FILE, sheet_name='cities_with_energy_data')
    cities = data_cities['City'].values
    latitudes = data_cities['latitude'].values
    longitudes = data_cities['longitude'].values
    climates = data_cities['climate'].values
    test_cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values
    scenarios = ["1990_2010", "B1_2100", "A2_2100", "A1B_2100"]

    flag_use_efficiency = True
    main(cities, latitudes, longitudes, climates, test_cities, scenarios, flag_use_efficiency)
