import os
import sys

sys.path.append(r'E:\GitHub\great-american-cities')

os.environ["THEANO_FLAGS"] = "device=cpu,floatX=float32,force_device=True"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import pickle
from sklearn.preprocessing import StandardScaler
import pymc3 as pm
import pandas as pd
import theano
from configuration import DATA_TRAINING_FILE, HIERARCHICAL_MODEL_INFERENCE_FOLDER_2_LEVELS


def main(Xy_training_path, output_trace_path, response_variable, predictor_variables, cities, samples, scaler_type):
    # READ DATA
    Xy_training = pd.read_csv(Xy_training_path)

    if cities != []:  # select cities to do the analysis
        Xy_training = Xy_training.loc[Xy_training['CITY'].isin(cities)]

    # SCALE THE DATA
    if scaler_type == "standard":
        scaler = StandardScaler()
        fields_to_scale = [response_variable] + predictor_variables
        Xy_training[fields_to_scale] = pd.DataFrame(scaler.fit_transform(Xy_training[fields_to_scale]),
                                                    columns=Xy_training[fields_to_scale].columns)
    else:
        scaler = None
        print("not scaling variables")

    # CREATE INDEXES FOR THE HIERACHY
    degree_index = Xy_training.groupby('CLIMATE_ZONE').all().reset_index().reset_index()[['index', 'CLIMATE_ZONE']]
    degree_state_index = Xy_training.groupby(["CLIMATE_ZONE", "CITY"]).all().reset_index().reset_index()[['index', "CLIMATE_ZONE", "CITY"]]
    degree_state_county_index = Xy_training.groupby(["CLIMATE_ZONE", "CITY", "BUILDING_CLASS"]).all().reset_index().reset_index()[['index', "CLIMATE_ZONE", "CITY", "BUILDING_CLASS"]]

    degree_state_indexes_df = pd.merge(degree_index, degree_state_index, how='inner', on='CLIMATE_ZONE', suffixes=('_d', '_ds'))
    degree_state_county_indexes_df = pd.merge(degree_state_indexes_df, degree_state_county_index, how='inner', on=['CLIMATE_ZONE', 'CITY'])
    indexed_salary_df = pd.merge(Xy_training, degree_state_county_indexes_df, how='inner', on=['CLIMATE_ZONE', 'CITY', 'BUILDING_CLASS']).reset_index()

    degree_indexes = degree_index['index'].values
    degree_count = len(degree_indexes)
    degree_state_indexes = degree_state_indexes_df['index_d'].values
    degree_state_count = len(degree_state_indexes)
    degree_state_county_indexes = degree_state_county_indexes_df['index_ds'].values
    degree_state_county_count = len(degree_state_county_indexes)

    # Xy_training[response_variable] = Xy_training[response_variable].astype(theano.config.floatX)

    with pm.Model() as hierarchical_model:

        # log(y) = alpha + beta*log(GFA*HDD) + gamma*log(GFA*CDD) + eps
        global_m = pm.Normal('global_m', mu=0, sd=10)
        global_m_sd = pm.Uniform('global_m_sd', lower=0, upper=10)
        global_b = pm.Normal('global_b', mu=0, sd=10)
        global_b_sd = pm.Uniform('global_b_sd', lower=0, upper=10)

        degree_m_offset = pm.Normal('degree_m_offset', mu=0, sd=10, shape=degree_count)
        degree_m = pm.Deterministic("degree_m", global_m + degree_m_offset * global_m_sd)

        degree_m_sd = pm.Uniform('degree_m_sd', lower=0, upper=10, shape=degree_count)

        b1_offset = pm.Normal('b1_offset', mu=0, sd=1, shape=n_counties)
        b1 = pm.Deterministic("b1", global_b1 + b1_offset * sigma_b1)
        #
        b2_offset = pm.Normal('b2_offset', mu=0, sd=1, shape=n_counties)
        b2 = pm.Deterministic("b2", global_b2 + b2_offset * sigma_b2)


        degree_b = pm.Normal('degree_b', mu=global_b, sd=global_b_sd, shape=degree_count)
        degree_b_sd = pm.Uniform('degree_b_sd', lower=0, upper=10, shape=degree_count)

        degree_state_m = pm.Normal('degree_state_m', mu=degree_m[degree_state_indexes], sd=degree_m_sd[degree_state_indexes], shape=degree_state_count)
        degree_state_m_sd = pm.Uniform('degree_state_m_sd', lower=0, upper=10, shape=degree_state_count)

        degree_state_b = pm.Normal('degree_state_b', mu=degree_b[degree_state_indexes],sd=degree_b_sd[degree_state_indexes], shape=degree_state_count)
        degree_state_b_sd = pm.Uniform('degree_state_b_sd', lower=0, upper=10, shape=degree_state_count)

        degree_state_county_m = pm.Normal('degree_state_county_m', mu=degree_state_m[degree_state_county_indexes],sd=degree_state_m_sd[degree_state_county_indexes],shape=degree_state_county_count)
        degree_state_county_b = pm.Normal('degree_state_county_b', mu=degree_state_b[degree_state_county_indexes],sd=degree_state_b_sd[degree_state_county_indexes],shape=degree_state_county_count)

        eps = pm.HalfCauchy('eps', 5)
        y_obs = indexed_salary_df[response_variable].values
        x1 = indexed_salary_df[predictor_variables[0]].values

        model = degree_state_county_m[indexed_salary_df['index'].values] * x1 + degree_state_county_b[indexed_salary_df['index'].values]

        # Data likelihood
        y_like = pm.Normal('y_like', mu=model, sd=eps, observed=y_obs)

    with hierarchical_model:

        step = pm.NUTS(target_accept=0.99)
        hierarchical_trace = pm.sample(draws=samples, step=step, n_init=samples, njobs=2)
        #  increase to avoid divergence problemsstep = pm.NUTS()  # increase to avoid divergence problems
        # hierarchical_trace = pm.sample(samples)
        # save to disc
        with open(output_trace_path, 'wb') as buff:
            pickle.dump({'inference': hierarchical_model, 'trace': hierarchical_trace,
                         'scaler': scaler, 'city_index_df': degree_state_county_indexes_df,
                         'response_variable': response_variable, 'predictor_variables': predictor_variables}, buff)


if __name__ == "__main__":
    scaler = "standard"  # "minmax"  # o standard for standard scaler to use in both sides of the data
    log_transform = "log_log"  # log_log or none
    type_log = "all_2var"  # "all" when the covariates and the response have log conversion, Floor if only floor is log (this behavior is modified manually)
    samples = 5000  # number of shamples per chain. Normally 2 chains are run so for 10.000 samples the sampler will do 20.0000 and compare convergence
    cities = []  # or leave empty to have all cities.
    response_variable = "LOG_SITE_ENERGY_kWh_yr"
    predictor_variables = ["LOG_THERMAL_ENERGY_kWh_yr"]

    Xy_training_path = DATA_TRAINING_FILE
    output_trace_path = os.path.join(HIERARCHICAL_MODEL_INFERENCE_FOLDER_2_LEVELS,
                                     log_transform + "_" + type_log + "_" + scaler + "_" + str(samples) + ".pkl")
    main(Xy_training_path, output_trace_path, response_variable, predictor_variables, cities, samples, scaler)
