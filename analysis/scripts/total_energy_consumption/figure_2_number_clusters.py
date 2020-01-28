from __future__ import division
from __future__ import print_function

import os

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from sklearn import mixture

from configuration import DATA_ALLDATA_FILE, DATA_CLUSTERING_PLOTS_FOLDER

random_state = 170
np.random.RandomState(random_state)
data_path = DATA_ALLDATA_FILE
df2 = pd.read_csv(data_path)
cities = df2.CITY.unique()
building_classes = df2.BUILDING_CLASS.unique()
aic_list = []
bic_list = []
city_cluster = []
building_class_cluster = []
for i, city in enumerate(cities):
    for j, classes in enumerate(building_classes):
        df3 = df2[df2["CITY"] == city]
        df3 = df3[df3["BUILDING_CLASS"] == classes]
        if df3.empty:
            x = 1
        else:
            X_cluster = df3[["LOG_SITE_EUI_kWh_m2yr"]].values
            cv_type = 'tied'
            n_componentssss = np.arange(1, 10)
            models = [mixture.GaussianMixture(n, covariance_type=cv_type, random_state=random_state).fit(X_cluster)
                      for n in n_componentssss]

            city_cluster.append(city)
            building_class_cluster.append(classes)
            aic_list.append(np.argmin(np.array([m.aic(X_cluster) for m in models])) + 1)
            # bic_list.append(np.argmin(np.array([m.bic(X_cluster) for m in models])) + 1)

# let's get the data for the number of clusters
# index 99 with has 5 clusters and it is the city of scaramento and Commercial buildings
titlex = r'$\log{x_{\text i,j}^{[1]}}$'
titley = r'$\log{y_{i,j}}$'
titlez = r'$\log{x_{\text i,j}^{[2]}}$'#r'$\log{x_{i,j}^{[2]}$'
n_components = 5
df3 = df2[df2["CITY"] == "New York, NY"]
df3 = df3[df3["BUILDING_CLASS"] == "Commercial"]
X_cluster = df3[["LOG_SITE_EUI_kWh_m2yr"]].values
cv_type = 'tied'
clusters = mixture.GaussianMixture(5, covariance_type=cv_type, random_state=random_state).fit(X_cluster)
gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
gmm.fit(X_cluster)
means = gmm.means_.T[0]  # /gmm.means_.T[1]
cluster_labels = gmm.predict(X_cluster)
df3['CLUSTERS'] = [round(means[cluster], 2) for cluster in cluster_labels]

xs = df3[["LOG_THERMAL_ENERGY_kWh_yr"]].values.flatten()
ys = df3[["LOG_SITE_ENERGY_kWh_yr"]].values.flatten()
zs = df3[["CLUSTERS"]].values.flatten()
trace = go.Scattergl(x=xs, y=ys,
                     mode='markers',
                     name='Pareto curve',
                     marker=dict(size=12, color=zs,
                                 colorscale='blues', showscale=True, opacity=0.8))
data = [trace]
layout = go.Layout(xaxis=dict(title=titlex),
                   font=dict(family='Helvetica, monospace', size=24, color='black'),
                   yaxis=dict(title=titley))

fig = go.Figure(data=data, layout=layout)
fig['layout'].update(plot_bgcolor="rgb(236,243,247)",
                     font=dict(family='Helvetica, monospace', size=24, color='black'),
                     annotations=[
        go.layout.Annotation(
            x=1.05,
            y=1.05,
            xref="paper",
            yref="paper",
            showarrow=False,
            text=titlez)])
outputfile = "clusters_example_one_city_and_class" + "_.html"
plot(fig, filename=os.path.join(DATA_CLUSTERING_PLOTS_FOLDER, outputfile), include_mathjax='cdn')

trace2 = go.Box(y=aic_list,
                x=['AIC'] * len(aic_list),
                name='AIC',
                boxmean=True,
                marker=dict(color="rgb(66,165,245)"))

data = [trace2]
layout = go.Layout(plot_bgcolor="rgb(236,243,247)",
                   legend=dict(x=0.90, y=0.95),
                   font=dict(family='Helvetica, monospace', size=24, color='black'),
                   xaxis=dict(title="Clustering Criterion", categoryorder="array", categoryarray=['AIC']),
                   yaxis=dict(title="Number of Clusters [-]", zeroline=False))
fig['layout'].update(plot_bgcolor="rgb(236,243,247)",
                     font=dict(family='Helvetica, monospace', size=24, color='black')
                     )
fig = go.Figure(data=data, layout=layout)

outputfile = "boxplot_clusters_criteria" + "_.html"
plot(fig, filename=os.path.join(DATA_CLUSTERING_PLOTS_FOLDER, outputfile))
