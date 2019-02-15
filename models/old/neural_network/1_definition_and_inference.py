import shutil
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

def build_model_columns(cities, building_classes):
  """Builds a set of wide and deep feature columns."""

  # Continuous columns
  HDD_FLOOR_18_5_C = tf.feature_column.numeric_column('LOG_HDD_FLOOR_18_5_C_m2')
  CDD_FLOOR_18_5_C = tf.feature_column.numeric_column('LOG_CDD_FLOOR_18_5_C_m2')

  # Categorical columns
  CITY = tf.feature_column.categorical_column_with_vocabulary_list('CITY', cities)
  # BUILDING_CLASS = tf.feature_column.categorical_column_with_vocabulary_list('BUILDING_CLASS', building_classes)

  # Wide columns and deep columns.
  wide_columns = [HDD_FLOOR_18_5_C, CDD_FLOOR_18_5_C, CITY]
  deep_columns = [HDD_FLOOR_18_5_C, CDD_FLOOR_18_5_C, tf.feature_column.indicator_column(CITY)] #tf.feature_column.indicator_column(BUILDING_CLASS)]

  return wide_columns, deep_columns


def build_estimator(model_dir, model_type, cities, building_classes, hidden_units):

  """Build an estimator appropriate for the given inference type."""
  wide_columns, deep_columns = build_model_columns(cities, building_classes)

  if model_type == 'wide':
    return tf.estimator.LinearRegressor(model_dir=model_dir,feature_columns=wide_columns)
                                        # config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNRegressor(model_dir=model_dir, feature_columns=deep_columns, hidden_units=hidden_units,
                                     )#activation_fn=tf.nn.sigmoid)

  else:
    return tf.estimator.DNNLinearCombinedRegressor(model_dir=model_dir,
                                                    linear_feature_columns=wide_columns,
                                                    dnn_feature_columns=deep_columns,
                                                    dnn_hidden_units=hidden_units)
                                                    #config=run_config)

def input_fn(data_file, FLAGS, scaler_X, scaler_y, train):
    """Generate an input function for the Estimator."""
    data = pd.read_csv(data_file)
    data = data[data['CITY'].isin(FLAGS['cities'])]
    data.reset_index(inplace=True)
    fields_to_scale = FLAGS["fields_to_scale"]

    y = data[FLAGS['response_variable']].values.reshape(-1, 1)
    if train:
        data[fields_to_scale] = pd.DataFrame(scaler_X.fit_transform(data[fields_to_scale]), columns=data[fields_to_scale].columns)
        labels = scaler_y.fit_transform(y)
    else:
        data[fields_to_scale] = pd.DataFrame(scaler_X.transform(data[fields_to_scale]),columns=data[fields_to_scale].columns)
        labels = scaler_y.transform(y.reshape(-1, 1))

    features = data[FLAGS['predictor_variables']]

    return features, labels, scaler_X, scaler_y

def train_fn(features, labels, FLAGS, buffer):
    # We call repeat after shuffling, rather than before, to prevent separate epochs from blending together.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(buffer_size=buffer).repeat(FLAGS['epoch_per_eval']).batch(FLAGS['batch_size']).make_one_shot_iterator().get_next()
    return dataset

def test_fn(features, labels, FLAGS):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)
    # We call repeat after shuffling, rather than before, to prevent separate epochs from blending together.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(FLAGS['batch_size']).make_one_shot_iterator().get_next()
    return dataset

def main(argv):

    # flags anc condiguration
    FLAGS = {}
    FLAGS['model_dir'] = os.path.join(os.getcwd(), "results", "inference", "log_neural_net_wide_deep_4L_453%_3%")
    FLAGS['model_type'] = ""
    FLAGS['num_epochs'] = 100
    FLAGS['epoch_per_eval'] = 2
    FLAGS['batch_size'] = 512 #use 32, 64, 128, 256, 512, 1024, 2048
    FLAGS['scaler_X'] = MinMaxScaler(feature_range=(-1, 1))
    FLAGS['scaler_y'] = MinMaxScaler(feature_range=(-1, 1))
    FLAGS['hidden_units'] = [100, 75, 50, 25]#
    FLAGS['response_variable'] = "LOG_SITE_ENERGY_MWh_yr"
    FLAGS['predictor_variables'] = ["LOG_HDD_FLOOR_18_5_C_m2", "LOG_CDD_FLOOR_18_5_C_m2", "CITY"] #"BUILDING_CLASS"]
    FLAGS["fields_to_scale"] = ["LOG_HDD_FLOOR_18_5_C_m2", "LOG_CDD_FLOOR_18_5_C_m2",]
    FLAGS['cities'] = pd.read_excel(os.path.join(os.getcwd(), "cities.xlsx"), sheet_name='cities_with_energy_data')['City'].values
    FLAGS['building_classes'] = ["Commercial", "Residential"]
        # ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Boston, MA", "Miami, FL", "Philadelphia, PA", "Washington, DC"]

    # indicate path to databases
    train_data_path = os.path.join(os.getcwd(), "data_processing", "data", "input_database.csv")
    test_data_path = os.path.join(os.getcwd(), "data_processing", "data", "testing_database.csv")

    # Upload data to memory and apply scaler of training data to the other variables
    X_train, y_train, FLAGS['scaler_X'], FLAGS['scaler_y'] = input_fn(train_data_path, FLAGS, FLAGS['scaler_X'], FLAGS['scaler_y'] , train = True)
    X_test, y_test, _, _ = input_fn(test_data_path, FLAGS, FLAGS['scaler_X'], FLAGS['scaler_y'], train = False)

    # # save flags and build inference

    with open(os.path.join(FLAGS['model_dir'],'flags.pkl'), 'wb') as fp:
        pickle.dump(FLAGS, fp)

    model = build_estimator(FLAGS['model_dir'], FLAGS['model_type'], FLAGS['cities'], FLAGS['building_classes'], FLAGS['hidden_units'])

    #  Clean up the inference directory if present
    shutil.rmtree(FLAGS['model_dir'], ignore_errors=True)
    # train the inference
    buffer_train = len(y_train)
    for n in range(FLAGS['num_epochs'] // FLAGS['epoch_per_eval']):
        model.train(input_fn=lambda: train_fn(X_train, y_train, FLAGS, buffer_train))
        results_test = model.evaluate(input_fn=lambda: test_fn(X_test, y_test, FLAGS), name="testing")

        # Display evaluation metrics
        print('Results at epoch', (n + 1) * FLAGS['epoch_per_eval'])
        print('-' * 60)

        for key in sorted(results_test):
            print('%s: %s' % (key, results_test[key]))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
