from __future__ import division
from __future__ import print_function

import plotly.graph_objs as go
from plotly import tools
import plotly.plotly as py

import os
from plotly.offline import plot
import pandas as pd
from configuration import DATA_POST_FUTURE_ENERGY_CONSUMPTION_FOLDER, DATA_ENERGY_PLOTS_FOLDER, NN_MODEL_PERFORMANCE_FOLDER, HIERARCHICAL_MODEL_PERFORMANCE_FOLDER, CONFIG_FILE
import numpy as np


def main(hierarchical_model_commercial,hierarchical_model_residential,neural_network_model):

    #get data from models to evaluate
    data_hierarchical_commercial = pd.read_csv(os.path.join(HIERARCHICAL_MODEL_PERFORMANCE_FOLDER, hierarchical_model_commercial + ".csv"))
    data_hierarchical_residential = pd.read_csv(os.path.join(HIERARCHICAL_MODEL_PERFORMANCE_FOLDER, hierarchical_model_residential + ".csv"))
    data_neural_network = pd.read_csv(os.path.join(NN_MODEL_PERFORMANCE_FOLDER, neural_network_model + ".csv"))
    data_hierarchical =  pd.concat([data_hierarchical_residential, data_hierarchical_commercial], ignore_index=True)
    data_neural_network["type"] = "Wide and Deep Neural Network"
    data_hierarchical["type"] = "Hierarchical Bayesian Linear Model"

    data_clean = pd.concat([data_hierarchical, data_neural_network], ignore_index=True)

    print(data_clean.head(10))

    graph_variable = ["Commercial",
                      "Residential",
                      "Commercial",
                      "Residential",
                      "Commercial",
                      "Residential"
                      ]

    quantities = ["MAPE_build_EUI_%", "MAPE_build_EUI_%",
                  "PE_mean_EUI_%", "PE_mean_EUI_%",
                  "MSE_log_domain", "MSE_log_domain"]

    titles = ["MAPE [%]", "MAPE [%]",
              "PE [%]", "PE [%]",
              "MSE (Log) [-]", "MSE (Log) [-]"]

    pairs_colors = [["rgb(66,165,245)", "rgb(13,71,161)"],
                    ["rgb(171,221,222)", "rgb(63,192,194)"],
                    ["rgb(66,165,245)", "rgb(13,71,161)"],
                    ["rgb(171,221,222)", "rgb(63,192,194)"],
                    ["rgb(66,165,245)", "rgb(13,71,161)"],
                    ["rgb(171,221,222)", "rgb(63,192,194)"]
                    ]

    # pairs_colors = [["rgb(144,202,249)", "rgb(66,165,245)", "rgb(13,71,161)"],
    #                 ["rgb(200,222,222)", "rgb(171,221,222)", "rgb(63,192,194)"]]

    #calculate training data
    data_train = data_clean[data_clean["DATASET"]== "Training"]
    data_test = data_clean[data_clean["DATASET"] == "Testing"]

    for grah_var, colors, title, quantity in zip(graph_variable, pairs_colors, titles, quantities):

        data_train_category = data_train[data_train["BUILDING_CLASS"] == grah_var]
        data_test_category = data_test[data_test["BUILDING_CLASS"] == grah_var]

        trace1 = go.Box(
            y= data_train_category[quantity],
            x= data_train_category["type"],
            name='Training data',
            boxpoints = 'all',
            marker=dict(color=colors[0]))

        trace2 = go.Box(
            y= data_test_category[quantity],
            x= data_test_category["type"],
            name='Testing data',
            boxpoints = 'all',
            marker=dict(color=colors[1]))

        data = [trace1, trace2]
        layout = go.Layout(plot_bgcolor= "rgb(236,243,247)",legend=dict(x=0.90, y=0.95),
                             font=dict(family='Helvetica, monospace', size=18),
            yaxis=dict(
                title=title,
                zeroline=False,
            ),
            boxmode='group',
        )
        fig = go.Figure(data=data, layout=layout)

        outputfile = "boxplot_percentage_error" + grah_var + "_" + quantity + "_.html"
        plot(fig, filename=os.path.join(DATA_ENERGY_PLOTS_FOLDER, outputfile))



if __name__ == "__main__":
    hierarchical_model_commercial = "log_logCommercial_all_2var_standard_5000"
    hierarchical_model_residential = "log_logResidential_all_2var_standard_5000"
    neural_network_model = "log_nn_wd_4L_2var"
    main(hierarchical_model_commercial,hierarchical_model_residential,neural_network_model)