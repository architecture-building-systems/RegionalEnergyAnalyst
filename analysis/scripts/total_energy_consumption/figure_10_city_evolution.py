from __future__ import division
from __future__ import print_function

import os

import math
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot

from configuration import DATA_ENERGY_PLOTS_FOLDER, \
    HIERARCHICAL_MODEL_PREDICTION_FOLDER_2_LEVELS_2_COVARIATE, CONFIG_FILE


def main(predictions_path, cities, output_path):
    # get the entire file
    final_df = pd.read_csv(os.path.join(predictions_path, "predictions_data.csv"))
    final_df = final_df[["CITY", "BUILDING_CLASS", "SCENARIO", "EUI_kWh_m2yr"]]
    final_df['SCENARIO_YEAR'] = [str(x.split('_')[-1]) for x in final_df['SCENARIO'].values]

    result = final_df.groupby(["CITY", "BUILDING_CLASS", "SCENARIO_YEAR"], as_index=False).agg(
        ['mean', 'count', 'std'])
    ci95_hi = []
    ci95_lo = []
    for i in result.index:
        m, c, s = result.loc[i]
        c = c/100 #because we used 100 samples from the bayesian to do this. in reality we have only 3 to 4 sources of uncertainty
        ci95_hi.append(m + 1.96 * s / math.sqrt(c))
        ci95_lo.append(m - 1.96 * s / math.sqrt(c))

    result['ci95_hi'] = ci95_hi
    result['ci95_lo'] = ci95_lo
    result = result.reset_index()

    data_clean = pd.DataFrame()
    data_clean['EUI_kWh_m2yr'] = result['EUI_kWh_m2yr']['mean']
    data_clean['CITY'] = result['CITY']
    data_clean['BUILDING_CLASS'] = result['BUILDING_CLASS']
    data_clean['SCENARIO_YEAR'] = result['SCENARIO_YEAR']
    data_clean['EUI_kWh_m2yr_min'] = result['ci95_lo']
    data_clean['EUI_kWh_m2yr_max'] = result['ci95_hi']

    for city in cities:
        data_city = data_clean[data_clean['CITY'] == city]
        if data_city.empty or data_city.empty:
            print(data_city, "does not exist, we are skipping it")
        else:
            data = []
            annotations = []
            for building_class, xpoint_annotation, color, fillcolor in zip(['Residential', 'Commercial'],
                                                                             ['2030', '2070'],
                                                                             ["rgb(13,71,161)", "rgb(63,192,194)"],
                                                                             ["rgba(144,202,249, 0.5)", "rgba(200,222,222,0.5)"]):

                data_class = data_city[data_city['BUILDING_CLASS'] == building_class]
                if data_class.empty or data_class.empty:
                    print(building_class, "does not exist, we are skipping it")
                else:

                    # annotation
                    Growth_rate_per_decade = (data_class.loc[
                                                  data_class['SCENARIO_YEAR'] == '2100', 'EUI_kWh_m2yr'].values[0]
                                              - data_class.loc[
                                                  data_class['SCENARIO_YEAR'] == '2010', 'EUI_kWh_m2yr'].values[0]) / 9
                    text_annotation = "GPD = " + str(round(Growth_rate_per_decade, 1))
                    y_point_annotation = data_class.loc[
                        data_class['SCENARIO_YEAR'] == xpoint_annotation, 'EUI_kWh_m2yr'].values[0]
                    upper_bound = go.Scatter(
                        name='Upper Bound' + building_class,
                        x=data_class['SCENARIO_YEAR'],
                        y=data_class['EUI_kWh_m2yr_max'],
                        mode='lines',
                        marker=dict(color=color),
                        line=dict(width=0),
                        fillcolor=fillcolor,
                        fill='tonexty')

                    trace = go.Scatter(
                        name='Measurement' + building_class,
                        x=data_class['SCENARIO_YEAR'],
                        y=data_class['EUI_kWh_m2yr'],
                        mode='lines',
                        line=dict(color=color),
                        fillcolor=fillcolor,
                        fill='tonexty')

                    lower_bound = go.Scatter(
                        name='Lower Bound' + building_class,
                        x=data_class['SCENARIO_YEAR'],
                        y=data_class['EUI_kWh_m2yr_min'],
                        marker=dict(color=color),
                        line=dict(width=0),
                        mode='lines')

                    # Trace order can be important
                    # with continuous error bars
                    data.extend([lower_bound, trace, upper_bound])
                    annotations.extend([
                        go.layout.Annotation(
                            x=xpoint_annotation,
                            y=y_point_annotation,
                            xref="x",
                            yref="y",
                            text=text_annotation,
                            showarrow=True,
                            arrowhead=7,
                            ax=0,
                            ay=-120)])
            layout = go.Layout(plot_bgcolor="rgb(236,243,247)",
                               xaxis=dict(title=data_city['CITY'].values[0], range=[2010, 2100]),
                               yaxis=dict(range=[0, 2000]),
                               font=dict(family='Helvetica, monospace', size=24, color='black'),
                               showlegend=False,
                               annotations=annotations
                               )

            fig = go.Figure(data=data, layout=layout)
            plot(fig, filename='pandas-continuous-error-bars')


if __name__ == "__main__":
    name_model = "log_log_all_2var_standard_2500"
    output_path = DATA_ENERGY_PLOTS_FOLDER
    predictions_path = os.path.join(HIERARCHICAL_MODEL_PREDICTION_FOLDER_2_LEVELS_2_COVARIATE, name_model)
    cities = pd.read_excel(CONFIG_FILE, sheet_name='test_cities')['City'].values
    main(predictions_path, cities, output_path)
