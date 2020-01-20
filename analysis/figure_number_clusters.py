from __future__ import division
from __future__ import print_function

import numpy as np
import numpy as np
import pandas as pd
from sklearn import mixture
import plotly.graph_objs as go
from plotly import tools
import plotly.plotly as py

import os
from plotly.offline import plot

from configuration import DATA_ALLDATA_FILE, DATA_CLUSTERING_PLOTS_FOLDER

random_state = 170
np.random.RandomState(random_state)
data_path = DATA_ALLDATA_FILE
df2 = pd.read_csv(data_path)
cities = df2.CITY.unique()
building_classes = df2.BUILDING_CLASS.unique()
aic_list = []
bic_list = []
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

            aic_list.append(np.argmin(np.array([m.aic(X_cluster) for m in models])) + 1)
            bic_list.append(np.argmin(np.array([m.bic(X_cluster) for m in models])) + 1)


trace1 = go.Box(y=bic_list,
                x=['BIC'] * len(aic_list),
                name='BIC',
                boxmean=True,
                marker=dict(color="rgb(239,83,80)"))

trace2 = go.Box(y=aic_list,
                x=['AIC'] * len(aic_list),
                name='AIC',
                boxmean=True,
                marker=dict(color="rgb(63,192,194)"))

data = [trace1, trace2]
layout = go.Layout(plot_bgcolor= "rgb(236,243,247)",
                   legend=dict(x=0.90, y=0.95),
                   font=dict(family='Helvetica, monospace', size=18),
                   xaxis=dict(title="Clustering Criterion", categoryorder="array", categoryarray=['BIC', 'AIC']),
                   yaxis=dict(title="Number of Clusters [-]", zeroline=False))

fig = go.Figure(data=data, layout=layout)

outputfile = "boxplot_clusters_criteria"+"_.html"
plot(fig, filename=os.path.join(DATA_CLUSTERING_PLOTS_FOLDER, outputfile))
