

import os
os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,force_device=True"
os.environ["MKL_THREADING_LAYER"] = "GNU"
import matplotlib.pyplot as plt
import numpy as np
import pickle
from configuration import HIERARCHICAL_MODEL_INFERENCE_FOLDER, HIERARCHICAL_MODEL_COEFFICIENT_PLOTS, CONFIG_FILE
import pymc3 as pm
import pandas as pd
import theano

def main(output_trace_path, output_graphs, main_cities):

    # loading data
    with open(output_trace_path, 'rb') as buff:
        data = pickle.load(buff)
        hierarchical_model, hierarchical_trace, scaler, degree_index, \
        response_variable, predictor_variables = data['inference'], data['trace'], data['scaler'], \
                                                 data['city_index_df'], data['response_variable'], \
                                                 data['predictor_variables']

    coefficients_plots(degree_index, hierarchical_trace, main_cities,output_graphs)

    # pm.traceplot(hierarchical_trace)
    # # pm.plot_posterior(hierarchical_trace, kde_plot =True)
    # pm.autocorrplot(hierarchical_trace, varnames=['global_a'])
    # bfmi = pm.bfmi(hierarchical_trace)
    # max_gr = max(np.max(gr_stats) for gr_stats in pm.gelman_rubin(hierarchical_trace).values())
    # (pm.energyplot(hierarchical_trace, legend=False, figsize=(6, 4)).set_title("BFMI = {} ;> 0.25 is ok \nGelman-Rubin ~ 1 = {} ; 1 < x < 1.2 is ok ".format(bfmi, max_gr)));
    plt.show()


def coefficients_plots(degree_index, hierarchical_trace, main_cities, output_graphs):
    # list of cities
    degree_index.set_index("CITY", inplace=True)
    # get data of traces
    data = pm.trace_to_dataframe(hierarchical_trace)
    alphas = []
    betas = []
    gamas = []
    for city in main_cities:
        i = degree_index.loc[city, "CODE"]
        alphas.append('alpha__' + str(i))
        betas.append('beta__' + str(i))
        gamas.append('gamma__' + str(i))
    data_alpha = data.rename(columns=dict(zip(alphas, main_cities)))
    data_alpha[main_cities].plot(kind='kde', title='b0',figsize=(7.5, 2.5), legend=False)
    plt.savefig(os.path.join(output_graphs, "b0.png"))

    data_beta = data.rename(columns=dict(zip(betas, main_cities)))
    data_beta[main_cities].plot(kind='kde', title='b1',figsize=(7.5, 2.5), legend=False)
    plt.savefig(os.path.join(output_graphs, "b1.png"))

    data_gama = data.rename(columns=dict(zip(gamas, main_cities)))
    data_gama[main_cities].plot(kind='kde', title='b2', figsize=(7.5, 2.5), legend=False)
    plt.savefig(os.path.join(output_graphs, "b2.png"))


if __name__ == "__main__":

    name_model = "log_log_all_standard_10000"
    output_trace_path = os.path.join(HIERARCHICAL_MODEL_INFERENCE_FOLDER, name_model + ".pkl")
    output_graphs = os.path.join(HIERARCHICAL_MODEL_COEFFICIENT_PLOTS, name_model)
    main_cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values
    main(output_trace_path, output_graphs, main_cities)