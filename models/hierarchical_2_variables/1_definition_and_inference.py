import os
import sys
sys.path.append(r'E:\GitHub\great-american-cities')

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
    Xy_training['BUILDING_CLASS'] = Xy_training['BUILDING_CLASS'].apply(lambda x: int(1) if x == "Residential" else int(0))

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

        # log(y) = alpha + beta*log(GFA*HDD) + gamma*log(GFA*CDD) + eps
        
        # Coefficients of all population
        global_b1 = pm.Normal('global_b1', mu=0., sd=100**2)
        sigma_b1 =  pm.HalfCauchy('sigma_b1', 5)

        global_b2 = pm.Normal('global_b2', mu=0., sd=100**2)
        sigma_b2 = pm.HalfCauchy('sigma_b2', 5)

        # Coefficients for each city, distributed around the group means
        b1 = pm.Normal('b1', mu=global_b1, sd=sigma_b1, shape=n_counties)
        b2 = pm.Normal('b2', mu=global_b2, sd=sigma_b2, shape=n_counties)

        # # Coefficients for each city, distributed around the group means
        # a_offset = pm.HalfCauchy('a_offset', 5, shape=n_counties)# pm.Normal('a_offset', mu=0, sd=10, shape=n_counties)
        # b1 = pm.Deterministic("b1", global_b1 + a_offset * sigma_b1)
        #
        # b_offset = pm.HalfCauchy('b_offset', 5, shape=n_counties)#pm.Normal('b_offset', mu=0, sd=10, shape=n_counties)
        # b2 = pm.Deterministic("b2", global_b2 + b_offset * sigma_b2)

        # Model error
        eps = pm.HalfCauchy('eps', 5)
        y_obs = Xy_training[response_variable]
        x1 = Xy_training[predictor_variables[0]].values

        model = b1[county_idx] + b2[county_idx] * x1

        # Data likelihood
        y_like = pm.Normal('y_like', mu=model, sd=eps, observed=y_obs)

    with hierarchical_model:

        step = pm.NUTS(target_accept=0.98)  # increase to avoid divergence problemsstep = pm.NUTS()  # increase to avoid divergence problems
        hierarchical_trace = pm.sample(draws=samples, step=step, n_init=samples, njobs=1)
        # save to disc
        with open(output_trace_path, 'wb') as buff:
            pickle.dump({'inference': hierarchical_model, 'trace': hierarchical_trace,
                         'scaler': scaler, 'city_index_df': degree_index,
                         'response_variable': response_variable, 'predictor_variables': predictor_variables}, buff)

if __name__ == "__main__":

    scaler = None #"standard" #"minmax"  # o standard for standard scaler to use in both sides of the data
    log_transform = "log_log" # log_log or none
    type_log = "all_2var"  # "all" when the covariates and the response have log conversion, Floor if only floor is log (this behavior is modified manually)
    samples = 2000 # number of shamples per chain. Normally 2 chains are run so for 10.000 samples the sampler will do 20.0000 and compare convergence
    cities = [] # or leave empty to have all cities.
    response_variable = "LOG_SITE_ENERGY_kWh_yr"
    predictor_variables = ["LOG_THERMAL_ENERGY_kWh_yr"]

    Xy_training_path = DATA_TRAINING_FILE
    output_trace_path = os.path.join(HIERARCHICAL_MODEL_INFERENCE_FOLDER,
                                     log_transform + "_" + type_log + "_" + "None" + "_" + str(samples) + ".pkl")
    main(Xy_training_path, output_trace_path, response_variable, predictor_variables, cities, samples, "None")
