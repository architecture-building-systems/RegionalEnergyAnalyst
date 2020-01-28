from __future__ import division
from __future__ import print_function

import os

import pandas as pd
import plotly.express as px

from configuration import NN_MODEL_PERFORMANCE_FOLDER, HIERARCHICAL_MODEL_PERFORMANCE_FOLDER_2_LEVELS_2_COVARIATE


def main(hierarchical_model_performance, neural_network_model_performance):
    # get data from models to evaluate
    data_hierarchical = pd.read_csv(os.path.join(HIERARCHICAL_MODEL_PERFORMANCE_FOLDER_2_LEVELS_2_COVARIATE,
                                                 hierarchical_model_performance + ".csv"))
    data_neural_network = pd.read_csv(
        os.path.join(NN_MODEL_PERFORMANCE_FOLDER, neural_network_model_performance + ".csv"))
    data_neural_network["TYPE"] = "WDNN"
    data_hierarchical["TYPE"] = "HBLM"

    # calculate graphs per building class
    building_classes = ['Commercial', 'Residential']
    for building_class in building_classes:
        data_hierarchical_clean = data_hierarchical[data_hierarchical["BUILDING_CLASS"] == building_class]
        data_neural_network_clean = data_neural_network[data_neural_network["BUILDING_CLASS"] == building_class]

        fig = px.histogram(data_hierarchical_clean,
                           x="MAPE_%",
                           y="CITY",
                           color="DATASET",
                           marginal="box",
                           color_discrete_sequence=["rgb(66,165,245)", "rgb(13,71,161)"])
        fig['layout'].update(plot_bgcolor="rgb(236,243,247)",
                             font=dict(family='Helvetica, monospace', size=24, color='black'))
        fig.show()
        fig = px.histogram(data_neural_network_clean,
                           x="MAPE_%",
                           y="CITY",
                           color="DATASET",
                           marginal="box",
                           color_discrete_sequence = ["rgb(66,165,245)", "rgb(13,71,161)"])
        fig['layout'].update(plot_bgcolor="rgb(236,243,247)",
                             font=dict(family='Helvetica, monospace', size=24, color='black'))
        fig.show()

    for building_class in building_classes:
        data_hierarchical_clean = data_hierarchical[data_hierarchical["BUILDING_CLASS"] == building_class]
        data_neural_network_clean = data_neural_network[data_neural_network["BUILDING_CLASS"] == building_class]

        fig = px.histogram(data_hierarchical_clean,
                           x="PE_%",
                           y="CITY",
                           color="DATASET",
                           marginal="box",
                           color_discrete_sequence=["rgb(66,165,245)", "rgb(13,71,161)"])
        fig['layout'].update(plot_bgcolor="rgb(236,243,247)",
                             font=dict(family='Helvetica, monospace', size=24, color='black'))
        fig.show()
        fig = px.histogram(data_neural_network_clean,
                           x="PE_%",
                           y="CITY",
                           color="DATASET",
                           marginal="box",
                           color_discrete_sequence = ["rgb(66,165,245)", "rgb(13,71,161)"])
        fig['layout'].update(plot_bgcolor="rgb(236,243,247)",
                             font=dict(family='Helvetica, monospace', size=24, color='black'))
        fig.show()

if __name__ == "__main__":
    hierarchical_model_performance = "log_log_all_2var_standard_2500"
    neural_network_model_performance = "log_nn_wd_4L_2var"
    main(hierarchical_model_performance, neural_network_model_performance)
