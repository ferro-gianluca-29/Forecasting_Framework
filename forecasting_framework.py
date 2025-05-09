#  LIBRARY IMPORTS

import argparse
import pandas as pd
import datetime
import pickle

from tools.data_preprocessing import DataPreprocessor
from tools.data_loader import DataLoader
from tools.performance_measurement import PerfMeasure
from tools.utilities import save_buffer, load_trained_model
from tools.time_series_analysis import time_s_analysis, multiple_STL

from Predictors.ARIMA_model import ARIMA_Predictor
from Predictors.SARIMA_model import SARIMA_Predictor
from Predictors.LSTM_torch import LSTM_Predictor
from Predictors.XGB_model import XGB_Predictor
from Predictors.NAIVE_model import NAIVE_Predictor


# END OF LIBRARY IMPORTS #

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

warnings.filterwarnings('ignore')


def main():
    """
    Main function to execute time series forecasting tasks based on user-specified arguments.
    This function handles the entire workflow from data loading, preprocessing, model training, testing,
    and evaluation, based on the configuration provided via command-line arguments.
    """

    # ARGUMENT PARSING
    """
    Parsing of command-line arguments to set up the environment and specify model training, testing, and evaluation parameters.
    """
    parser = argparse.ArgumentParser(description='Time series forecasting')

    # General arguments
    parser.add_argument('--verbose', action='store_true', required=False, default=False,
                        help='If specified, minimizes the additional information provided during the program launch')
    parser.add_argument('--ts_analysis', action='store_true', required=False, default=False,
                        help='If True, performs an analysis on the time series')
    parser.add_argument('--run_mode', type=str, required=True,
                        help='Running mode (training, testing, both, or fine tuning)')

    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset path')
    parser.add_argument('--date_format', type=str, required=True, help='Format of date time')
    parser.add_argument('--date_list', type=str, nargs='+',
                        help='List with start and end of dates for training, validation and test set')
    parser.add_argument('--train_size', type=float, required=False, default=0.7, help='Training set size')
    parser.add_argument('--val_size', type=float, required=False, default=0.2, help='Validation set size')
    parser.add_argument('--test_size', type=float, required=False, default=0.1, help='Test set size')
    parser.add_argument('--scaling', action='store_true', help='If True, data will be scaled')
    parser.add_argument('--validation', action='store_true', required=False,
                        help='If True, the validation set is created')
    parser.add_argument('--target_column', type=str, required=True, help='Name of the target column for forecasting')
    parser.add_argument('--time_column_index', type=int, required=False, default=0,
                        help='Index of the column containing the timestamps')
    parser.add_argument('--data_freq', type=str, required=False,
                        help='Time frequency of dataset. Required for XGB model')

    # Model arguments
    parser.add_argument('--model_type', type=str, required=True, help='Type of model to use (ARIMA, SARIMA, LSTM, XGB)')

    # Statistical models
    parser.add_argument('--forecast_type', type=str, required=False,
                        help='Type of forecast: ol-multi= open-loop multi step ahead; ol-one= open loop one step ahead, cl-multi= closed-loop multi step ahead')
    parser.add_argument('--valid_steps', type=int, required=False, default=10,
                        help='Number of time steps to use during validation')
    parser.add_argument('--steps_jump', type=int, required=False, default=50,
                        help='Number of steps to skip in open loop multi step predictions')
    parser.add_argument('--exog', nargs='+', type=str, required=False, default=None,
                        help='Exogenous columns for the SARIMAX model')
    parser.add_argument('--period', type=int, required=False, default=24, help='Seasonality period')
    parser.add_argument('--set_fourier', action='store_true', required=False, default=False,
                        help='If True, Fourier exogenous variables are used')

    # Other models
    parser.add_argument('--seasonal_model', action='store_true',
                        help='If True, in the case of LSTM the seasonal component is fed into the model, while for XGB models Fourier features are added')
    parser.add_argument('--input_len', type=int, required=False, default=24,
                        help='Number of timesteps to use for prediction in each window in LSTM')
    parser.add_argument('--output_len', type=int, required=False, default=1,
                        help='Number of timesteps to predict in each window in LSTM')

    # Test and fine tuning arguments
    parser.add_argument('--model_path', type=str, required=False, default=None, help='Path of the pre-trained model')
    parser.add_argument('--ol_refit', action='store_true', required=False, default=False,
                        help='For ARIMA and SARIMAX models: If specified, in OL forecasts the model is retrained for each added observation ')
    parser.add_argument('--unscale_predictions', action='store_true', required=False, default=False,
                        help=' If specified, predictions and test data are unscaled')

    args = parser.parse_args()
    # END OF ARGUMENT PARSING

    verbose = args.verbose

    try:

        # Create current model folder
        folder_name = args.model_type + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_path = f"./data/models/{folder_name}"
        os.makedirs(folder_path)

        #######  DATA LOADING

        data_loader = DataLoader(args.dataset_path, args.date_format, args.model_type, args.target_column,
                                 args.time_column_index, args.date_list, args.exog)
        df, dates = data_loader.load_data()
        if df is None:
            raise ValueError("Unable to load dataset.")

        ####### END OF DATA LOADING

        ####### PREPROCESSING AND DATASET SPLIT  ########

        # Extract the file extension from the path
        file_ext = os.path.splitext(args.dataset_path)[1]

        data_preprocessor = DataPreprocessor(file_ext, args.run_mode, args.model_type, df, args.target_column, dates,
                                             args.scaling, args.validation, args.train_size, args.val_size,
                                             args.test_size,
                                             folder_path, args.model_path, verbose)

        ############### Optional time series analysis ############

        if args.ts_analysis:
            time_s_analysis(df, args.target_column, args.period, d=1, D=0)
            train, test, exit = data_preprocessor.preprocess_data()

            #multiple_STL(train, args.target_column)
            return 0

        ############## End of time series analysis ###########

        #### Model Selection ####

        match args.model_type:

            case 'ARIMA':
                arima = ARIMA_Predictor(args.run_mode, args.target_column,
                                        args.verbose)

            case 'SARIMA':
                sarima = SARIMA_Predictor(args.run_mode, args.target_column, args.period,
                                          args.verbose)

            case 'LSTM':
                lstm = LSTM_Predictor(args.run_mode, args.target_column,
                                      args.verbose, args.input_len, args.output_len, args.validation)

            case 'XGB':
                xgb = XGB_Predictor(args.run_mode, args.target_column,
                                    args.verbose, args.seasonal_model, args.input_len, args.output_len,
                                    args.forecast_type, args.period)


            case 'NAIVE':
                naive = NAIVE_Predictor(args.run_mode, args.target_column,
                                        args.verbose)

        #### End of model selection ####

        ### Preprocessing for test-only mode
        if args.run_mode == "test":

            test, exit = data_preprocessor.preprocess_data()

            match args.model_type:

                case 'ARIMA':
                    arima.prepare_data(test=test)

                case 'SARIMA' | 'SARIMAX':

                    # Create the test set for target and exog variables
                    target_test = test[[args.target_column]]
                    if args.exog is not None:
                        exog_test = test[args.exog]
                    else:
                        exog_test = None
                    sarima.prepare_data(test=target_test)

                case 'LSTM':
                    train = []
                    valid = []
                    lstm.prepare_data(train, valid, test)

                case 'XGB':
                    train = []
                    valid = []
                    xgb.prepare_data(train, valid, test)

            ### End of preprocessing for test-only mode

        else:

            ####### PREPROCESSING AND DATASET SPLIT ######
            match args.model_type:

                case 'ARIMA':
                    if args.validation:
                        train, test, valid, exit = data_preprocessor.preprocess_data()
                        if exit:
                            raise ValueError("Unable to preprocess dataset.")

                    else:
                        train, test, exit = data_preprocessor.preprocess_data()
                        valid = None

                        if exit:
                            raise ValueError("Unable to preprocess dataset.")

                    arima.prepare_data(train, test)

                case 'SARIMA' | 'SARIMAX':
                    # Set the exogenous variable column
                    if args.exog is None:
                        exog = args.target_column
                    else:
                        exog = args.exog

                    if args.validation:
                        train, test, valid, exit = data_preprocessor.preprocess_data()
                        if exit:
                            raise ValueError("Unable to preprocess dataset.")
                        target_valid = valid[[args.target_column]]
                        exog_valid = valid[exog]


                    else:
                        train, test, exit = data_preprocessor.preprocess_data()
                        if exit:
                            raise ValueError("Unable to preprocess dataset.")
                        valid = None
                        exog_valid = None

                    target_train = train[[args.target_column]]
                    exog_train = train[exog]

                    if args.run_mode == 'train_test':
                        target_test = test[[args.target_column]]
                        exog_test = test[exog]
                    else:
                        target_test = None

                    sarima.prepare_data(train, test)

                case 'LSTM':
                    train, test, exit = data_preprocessor.preprocess_data()
                    if exit:
                        raise ValueError("Unable to preprocess dataset.")

                    lstm.prepare_data(train, test)

                case 'XGB':

                    train, test, exit = data_preprocessor.preprocess_data()
                    if exit:
                        raise ValueError("Unable to preprocess dataset.")

                    xgb.prepare_data(train, test)


                case 'NAIVE':
                    train, test, exit = data_preprocessor.preprocess_data()
                    valid = None
                    model = None
                    if exit:
                        raise ValueError("Unable to preprocess dataset.")
                    naive.prepare_data(train, None, test)

            print(f"Training set dim: {train.shape[0]} \n")
            if args.run_mode == 'train_test': print(f"Test set dim: {test.shape[0]}")

        ########### END OF PREPROCESSING AND DATASET SPLIT ########

        if args.run_mode == "train" or args.run_mode == "train_test":

            #################### MODEL TRAINING ####################

            match args.model_type:

                case 'ARIMA':
                    model, valid_metrics, last_index = arima.train_model()
                    best_order = arima.ARIMA_order
                    if args.run_mode == 'train_test':
                        # Save a buffer containing the last elements of the training set for further test
                        buffer_size = test.shape[0]
                        save_buffer(folder_path, train, args.target_column, size=buffer_size, file_name='buffer.json')
                    # Save the model
                    #...

                case 'SARIMAX' | 'SARIMA':
                    model, valid_metrics, last_index = sarima.train_model()
                    best_order = sarima.SARIMA_order
                    if args.run_mode == 'train_test':
                        # Save a buffer containing the last elements of the training set for further test
                        buffer_size = test.shape[0]
                        save_buffer(folder_path, train, args.target_column, size=buffer_size, file_name='buffer.json')
                    # Save the model
                    #...

                case 'LSTM':
                    model = lstm.train_model()
                    # Save the model
                    # ...

                case 'XGB':

                    model = xgb.train_model()
                    # model = xgb.hyperparameter_tuning()

                    # Save the model
                    # ...


            #################### END OF MODEL TRAINING ####################

        if args.run_mode == "train_test" or args.run_mode == "fine_tuning" or args.run_mode == "test":

            ##### Manage buffer for statistical models
            if args.run_mode == "test" and args.model_type in ['ARIMA', 'SARIMA', 'SARIMAX']:
                # Create a training set from the loaded buffer, that will be used for the naive models
                # for ARIMA: train; for SARIMAX: target_train
                # Load buffer from JSON file
                train = pd.DataFrame()
                train[args.target_column] = pd.read_json(f"{args.model_path}/buffer.json", orient='records')
                target_train = pd.DataFrame()
                target_train[args.target_column] = pd.read_json(f"{args.model_path}/buffer.json", orient='records')
                print(f"Training set buffer dim: {train.shape[0]} \n")
                print(f"Test set dim: {test.shape[0]}")
            ##### End of manage buffer

            #################### MODEL TESTING ####################

            match args.model_type:

                case 'ARIMA':
                    # Model testing

                    predictions = arima.test_model(model, args.forecast_type, args.output_len, args.ol_refit)

                    if args.unscale_predictions:

                        if args.run_mode == 'train_test':
                            path = folder_path
                        else:
                            path = args.model_path

                        # Load scaler for unscaling data
                        with open(f"{folder_path}/scaler.pkl", "rb") as file:
                            scaler = pickle.load(file)
                        predictions[args.target_column] = scaler.inverse_transform(predictions[[args.target_column]])

                        # Unscale test data
                        # Load scaler for unscaling test data
                        with open(f"{path}/scaler.pkl", "rb") as file:
                            scaler = pickle.load(file)
                        test[args.target_column] = scaler.inverse_transform(test[[args.target_column]])

                    predictions.to_csv('raw_data.csv', index=False)

                case 'SARIMAX' | 'SARIMA':
                    # Model testing
                    predictions = sarima.test_model(model, args.forecast_type, args.output_len, args.ol_refit)

                    if args.unscale_predictions:

                        if args.run_mode == 'train_test':
                            path = folder_path
                        else:
                            path = args.model_path

                        # Load scaler for unscaling data
                        with open(f"{folder_path}/scaler.pkl", "rb") as file:
                            scaler = pickle.load(file)
                        predictions[args.target_column] = scaler.inverse_transform(predictions[[args.target_column]])

                        # Unscale test data
                        # Load scaler for unscaling test data
                        with open(f"{path}/scaler.pkl", "rb") as file:
                            scaler = pickle.load(file)
                        test[args.target_column] = scaler.inverse_transform(test[[args.target_column]])

                    predictions.to_csv('raw_data.csv', index=False)

                case 'LSTM':
                    # Model testing

                    predictions, y_test = lstm.test_model(model)
                    if args.unscale_predictions:

                        if args.run_mode == 'train_test':
                            path = folder_path
                        else:
                            path = args.model_path

                        # Load scaler for unscaling data
                        with open(f"{folder_path}/scaler.pkl", "rb") as file:
                            scaler = pickle.load(file)
                        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

                        # Unscale test data
                        # Load scaler for unscaling test data
                        with open(f"{path}/scaler.pkl", "rb") as file:
                            scaler = pickle.load(file)
                        test[args.target_column] = scaler.inverse_transform(test[[args.target_column]])
                        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

                    pd.Series(predictions.flatten()).to_csv('raw_data.csv', index=False)

                case 'XGB':

                    # Model testing

                    predictions = xgb.test_model(model)

                    if args.unscale_predictions:

                        if args.run_mode == 'train_test':
                            path = folder_path
                        else:
                            path = args.model_path

                        # Load scaler for unscaling data
                        with open(f"{folder_path}/scaler.pkl", "rb") as file:
                            scaler = pickle.load(file)
                        predictions[args.target_column] = scaler.inverse_transform(predictions[[args.target_column]])

                        # Unscale test data
                        # Load scaler for unscaling test data
                        with open(f"{path}/scaler.pkl", "rb") as file:
                            scaler = pickle.load(file)
                        test[args.target_column] = scaler.inverse_transform(test[[args.target_column]])

                    predictions = predictions[args.target_column]

                    pd.Series(predictions).to_csv('raw_data.csv', index=False)


                case 'NAIVE':
                    if args.seasonal_model:
                        predictions = naive.seasonal_forecast(args.period)
                    else:
                        predictions = naive.forecast(args.forecast_type)

                    if args.unscale_predictions:
                        # Unscale predictions

                        predictions = naive.unscale_predictions(predictions, folder_path)

                        # Unscale test data
                        # Load scaler for unscaling test data
                        with open(f"{folder_path}/scaler.pkl", "rb") as file:
                            scaler = pickle.load(file)

                        test[args.target_column] = scaler.inverse_transform(test[[args.target_column]])

                    predictions.to_csv('raw_data.csv', index=False)

            #################### END OF MODEL TESTING ####################

        if args.run_mode != "train":

            #################### PLOT PREDICTIONS ####################

            match args.model_type:

                case 'ARIMA':
                    arima.plot_predictions(predictions)

                case 'SARIMAX' | 'SARIMA':
                    sarima.plot_predictions(predictions)

                case 'LSTM':
                    lstm.plot_predictions(predictions, y_test)

                case 'XGB':
                    xgb.plot_predictions(predictions)

                case 'NAIVE':
                    naive.plot_predictions(predictions)

                    #################### END OF PLOT PREDICTIONS ####################

            #################### PERFORMANCE MEASUREMENT AND SAVING #################

            perf_measure = PerfMeasure(args.model_type, test, args.target_column, args.forecast_type)

            match args.model_type:

                case 'ARIMA':
                    # Compute performance metrics
                    metrics = perf_measure.get_performance_metrics(test[args.target_column],
                                                                   predictions[args.target_column])
                    # Save metrics
                    arima.save_metrics(folder_path, metrics)

                case 'SARIMAX' | 'SARIMA':
                    # Compute performance metrics
                    metrics = perf_measure.get_performance_metrics(target_test, predictions)
                    # Save model data
                    sarima.save_metrics(folder_path, metrics)

                case 'LSTM':
                    # Compute performance metrics
                    metrics = perf_measure.get_performance_metrics(y_test, predictions)
                    # Save metrics
                    lstm.save_metrics(folder_path, metrics)

                case 'XGB':
                    # Compute performance metrics
                    metrics = perf_measure.get_performance_metrics(test[args.target_column], predictions)
                    # Save metrics
                    xgb.save_metrics(folder_path, metrics)

                case 'NAIVE':
                    # Save metrics
                    metrics = perf_measure.get_performance_metrics(test[args.target_column], predictions)
                    naive.save_metrics(folder_path, metrics)

            #################### END OF PERFORMANCE MEASUREMENT AND SAVING ####################


    except Exception as e:
        print(f"An error occurred in Main: {e}")


if __name__ == "__main__":
    main()
