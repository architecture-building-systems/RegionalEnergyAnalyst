import os

os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,force_device=True"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pymc3 as pm
import pandas as pd
import theano
from configuration import DATA_TRAINING_FILE, HIERARCHICAL_MODEL_INFERENCE_FOLDER

def main(Xy_training_path, output_trace_path, response_variable, predictor_variables, cities, samples, scaler_type):
    Xy_training = pd.read_csv(Xy_training_path)

    if cities != []:  # select cities to do the analysis
        Xy_training = Xy_training.loc[Xy_training['CITY'].isin(cities)]

    degree_index = Xy_training.groupby('CITY').all().reset_index().reset_index()[['index', 'CITY']]
    degree_index["CODE"] = degree_index.index.values
    Xy_training = Xy_training.merge(degree_index, on='CITY')

    if scaler_type == "minmax":
        scaler = MinMaxScaler(feature_range=(0.1, 0.99))
        fields_to_scale = [response_variable] + predictor_variables
        Xy_training[fields_to_scale] = pd.DataFrame(scaler.fit_transform(Xy_training[fields_to_scale]),
                                                    columns=Xy_training[fields_to_scale].columns)
    elif scaler_type == "minmax_extended":
        scaler = MinMaxScaler(feature_range=(-0.99, 0.99))
        fields_to_scale = [response_variable] + predictor_variables
        Xy_training[fields_to_scale] = pd.DataFrame(scaler.fit_transform(Xy_training[fields_to_scale]),
                                                    columns=Xy_training[fields_to_scale].columns)
    elif scaler_type =="standard":
        scaler = StandardScaler()
        fields_to_scale = [response_variable] + predictor_variables
        Xy_training[fields_to_scale] = pd.DataFrame(scaler.fit_transform(Xy_training[fields_to_scale]),
                                                    columns=Xy_training[fields_to_scale].columns)
    else:
        scaler = None
        print("not scaling variables")

    Xy_training[response_variable] = Xy_training[response_variable].astype(theano.config.floatX)

    mn_counties = Xy_training.CITY.unique()
    n_counties = len(mn_counties)
    county_idx = Xy_training.CODE.values

    with pm.Model() as hierarchical_model:

        # log(y) = a + b*log(floor) + g*HDD + e*CDD
        # Hyperpriors for group nodes
        global_a = pm.Uniform('global_a', lower=0., upper=5) # pm.Normal('global_a', mu=0., sd=100 ** 2)
        sigma_a = pm.HalfCauchy('sigma_a', 5)

        global_b = pm.Uniform('global_b', lower=0., upper=5) # pm.Normal('global_b', mu=0., sd=100 ** 2)
        sigma_b = pm.HalfCauchy('sigma_b', 5)

        global_c = pm.Uniform('global_c', lower=0., upper=5) # pm.Normal('global_c', mu=0., sd=100 ** 2) #pm.StudentT('global_d', nu = 5, mu=0, sd=100**2)
        sigma_c = pm.HalfCauchy('sigma_c', 5)

        # Intercept for each county, distributed around group mean mu_a
        a_offset = pm.HalfCauchy('a_offset', 5, shape=n_counties)# pm.Normal('a_offset', mu=0, sd=10, shape=n_counties)
        alpha = pm.Deterministic("alpha", global_a + a_offset * sigma_a)

        b_offset = pm.HalfCauchy('b_offset', 5, shape=n_counties)#pm.Normal('b_offset', mu=0, sd=10, shape=n_counties)
        beta = pm.Deterministic("beta", global_b + b_offset * sigma_b)

        c_offset = pm.HalfCauchy('c_offset', 5, shape=n_counties) #pm.Normal('c_offset', mu=0, sd=10, shape=n_counties)
        gamma = pm.Deterministic("gamma", global_c + c_offset * sigma_c)

        # Model error
        eps = pm.HalfCauchy('eps', 5)
        y_obs = Xy_training[response_variable]
        x1 = Xy_training[predictor_variables[0]].values
        x2 = Xy_training[predictor_variables[1]].values
        model = alpha[county_idx] + beta[county_idx] * x1 + gamma[county_idx] * x2

        # Data likelihood
        y_obs = pm.Normal('y_obs', mu=model, sd=eps, observed=y_obs)

    with hierarchical_model:

        step = pm.NUTS(target_accept=0.95)  # increase to avoid divergence problems
        hierarchical_trace = pm.sample(draws=samples, step=step, n_init=1000, njobs=2)
        # save to disc
        with open(output_trace_path, 'wb') as buff:
            pickle.dump({'inference': hierarchical_model, 'trace': hierarchical_trace,
                         'scaler': scaler, 'city_index_df': degree_index,
                         'response_variable': response_variable, 'predictor_variables': predictor_variables}, buff)

if __name__ == "__main__":

    scaler = "standard" #"minmax"  # o standard for standard scaler to use in both sides of the data
    log_transform = "log_log" # log_log or none
    type_log = "all"  # "all" when the covariates and the response have log conversion, Floor if only floor is log (this behavior is modified manually)
    samples = 10000 # number of shamples per chain. Normally 2 chains are run so for 10.000 samples the sampler will do 20.0000 and compare convergence
    cities = [] # or leave empty to have all cities.
    response_variable = "LOG_SITE_ENERGY_MWh_yr"
    predictor_variables = ["LOG_HDD_FLOOR_18_5_C_m2", "LOG_CDD_FLOOR_18_5_C_m2"]

    Xy_training_path = DATA_TRAINING_FILE
    output_trace_path = os.path.join(HIERARCHICAL_MODEL_INFERENCE_FOLDER,
                                     log_transform + "_" + type_log + "_" + scaler + "_" + str(samples) + ".pkl")
    main(Xy_training_path, output_trace_path, response_variable, predictor_variables, cities, samples, scaler)
