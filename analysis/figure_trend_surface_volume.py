from configuration import CONFIG_FILE,  DATA_ANALYSIS_PLOTS_FOLDER, DATA_RAW_BUILDING_GEOMETRY_FOLDER
import pandas as pd
import os
from geopandas import GeoDataFrame as Gdf
import matplotlib.pyplot as plt

cities = pd.read_excel(CONFIG_FILE, sheet_name='cities_building_performance')['City']
data_path = os.getcwd()
output_path = os.path.join(DATA_ANALYSIS_PLOTS_FOLDER, "trend_surface_to_GFA.jpeg")


for i, city in enumerate(cities):
    print(city)
    buildings_geometry = Gdf.from_file(os.path.join(DATA_RAW_BUILDING_GEOMETRY_FOLDER, city +".dbf"))
    buildings_geometry.plot.line(x="S_m2", y="GFA_m2")
    plt.show()
    x= 1
