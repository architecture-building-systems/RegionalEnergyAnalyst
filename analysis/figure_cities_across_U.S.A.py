from __future__ import division
from __future__ import print_function

import plotly.graph_objs as go
import os
from plotly.offline import plot
import pandas as pd
from configuration import DATA_ANALYSIS_PLOTS_FOLDER, HIERARCHICAL_MODEL_PREDICTION_FOLDER

def load_data():
    import os
    import pandas as pd
    from configuration import CONFIG_FILE, DATA_RAW_BUILDING_PERFORMANCE_FOLDER

    # get input databases
    cities_energy_data = pd.read_excel(CONFIG_FILE, sheet_name="cities_with_energy_data")
    cities_location_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_us_cities.csv')

    # clean energy data city names
    cities_energy_data["name"] = [x.split(',')[0] for x in cities_energy_data['City']]
    cities_location_data["name"] = [x.rstrip() for x in cities_location_data['name']]
    cities_location_data["name"] = [x if x != "St. Paul" else "Saint Paul" for x in cities_location_data['name']]
    cities_location_data["name"] = [x if x != "Nashville-Davidson" else "Nashville" for x in
                                    cities_location_data['name']]
    cities_location_data["name"] = [x if x != "Columbia Heights" else "Columbia" for x in cities_location_data['name']]
    cities_location_data["name"] = [x if x != "Boise City" else "Boise" for x in cities_location_data['name']]
    cities_location_data["name"] = [x if x != "North Augusta" else "Augusta" for x in cities_location_data['name']]
    # cities_energy_data.set_index(cities_energy_data["name"], inplace=True)
    # cities_location_data.set_index(cities_location_data["name"], inplace=True)
    data_join = pd.merge(cities_energy_data, cities_location_data, on="name")
    # pd.merge(cities_energy_data, cities_location_data, left_on="name", right_on="name", how="outer")

    data_join["number_records"] = 0
    lenght_records = len(data_join["number_records"].values)
    for i in range(lenght_records):
        a = data_join.loc[i, "City"]
        read_values = pd.read_csv(os.path.join(DATA_RAW_BUILDING_PERFORMANCE_FOLDER, data_join.loc[i, "City"] + ".csv"))
        data_join.loc[i, "number_records"] = len(read_values.area.values)

    return data_join, lenght_records


df, records=load_data()
output_path = os.path.join(DATA_ANALYSIS_PLOTS_FOLDER, "data_energy_consumption_cities.html")

df['text'] = df['name'] + '<br>Records ' + (df['number_records']).astype(str)
limits = [(0,1000),(1001, 5000),(5001,10000),(10001,20000),(20001,40000)]
colors = ["rgb(63,192,194)","rgb(255,209,29)","rgb(126,199,143)","rgb(245,131,69)","rgb(240,75,91)"]
cities = []
scale = 5

data_splitted = []
df["limits"] = 0
for i, lim in enumerate(limits):
    for j in range(records):
        if lim[0] < df.loc[j, 'number_records'] <= lim[1]:
            df.loc[j, 'limits'] = i

for i in range(len(limits)):
    df_sub = df[df['limits'] == i]
    lim = limits[i]
    city = dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = df_sub['lon'],
        lat = df_sub['lat'],
        text = df_sub['text'],
        marker = dict(
            size = df_sub['number_records']/scale,
            color = colors[i],
            line = dict(width=0.5, color='rgb(40,40,40)'),
            sizemode = 'area'
        ),
        name = '{0} - {1}'.format(lim[0],lim[1]))
    cities.append(city)

layout = dict(
        title = '2014 US city populations<br>(Click legend to toggle traces)',
        showlegend = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = 'rgb(255, 255, 255)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        ),
    )

fig = dict( data=cities, layout=layout )
plot(fig, validate=False, filename=output_path)

