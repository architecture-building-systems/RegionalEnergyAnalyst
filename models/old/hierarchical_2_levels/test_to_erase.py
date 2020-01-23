import pandas as pd
import numpy as np

set_parameters = (
    (('no degree', 'CA', 'Riverside County'), (5000, 1250)),
    (('no degree', 'IL', 'Cook County'), (6500, 1150)),
    (('no degree', 'IL', 'Lake County'), (7000, 1350)),

    (('degree', 'CA', 'Riverside County'), (6000, 1250)),
    (('degree', 'IL', 'Cook County'), (7500, 1150)),
    (('degree', 'IL', 'Lake County'), (8000, 1350))
)

# Go through each definition above and generate a fake time series
rows = []
data_points = [2000]
for set_parameter, N in zip(set_parameters, data_points):
    key, parameters = set_parameter
    population_rows = [{
        'degree': key[0],
        'state': key[1],
        'county': key[2],
        'month_index': i,
        # Create time series data and add some noise to make it realistic
        'salary': i * parameters[0] + parameters[1] + np.random.normal(loc=0, scale=500)
    } for i in range(N)]
    rows += population_rows

salary_df = pd.DataFrame(rows)
print (salary_df)