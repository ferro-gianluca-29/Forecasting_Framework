from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import datetime as datetime
import pickle

class DataPreprocessor():
    """
    A class to handle operations of preprocessing, including tasks such as managing NaN values,
    removing non-numeric columns, splitting datasets, managing outliers, and scaling data.

    :param file_ext: File extension for saving datasets.
    :param run_mode: Mode of operation ('train', 'test', 'train_test', 'fine_tuning').
    :param model_type: Type of machine learning model to prepare data for.
    :param df: DataFrame containing the data.
    :param target_column: Name of the target column in the DataFrame.
    :param dates: Indexes of dates given by command line with --date_list.
    :param scaling: Boolean flag to determine if scaling should be applied.
    :param validation: Boolean flag to determine if a validation set should be created.
    :param train_size: Proportion of data to be used for training.
    :param val_size: Proportion of data to be used for validation.
    :param test_size: Proportion of data to be used for testing.
    :param folder_path: Path to folder for saving data.
    :param model_path: Path to model file for loading or saving the model.
    :param verbose: Boolean flag for verbose output.
    """    
    def __init__(self, file_ext, run_mode, model_type, df: pd.DataFrame, target_column: str, dates = None, 
                 scaling = False, validation = None, train_size = 0.7, val_size = 0.2, test_size = 0.1, 
                 folder_path = None, model_path = None,  verbose = False):
        
        self.file_ext = file_ext
        self.run_mode = run_mode
        self.dates = dates
        self.model_type = model_type
        self.df = df
        self.target_column = target_column
        self.target_column_index = self.df.columns.get_loc(target_column)
        self.scaling = scaling
        self.validation = validation
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.folder_path = folder_path
        self.model_path = model_path
        self.verbose = verbose

    def conditional_print(self, *args, **kwargs):
        """
        Print messages conditionally based on the verbose attribute.

        :param args: Non-keyword arguments to be printed
        :param kwargs: Keyword arguments to be printed
        """
        if self.verbose:
            print(*args, **kwargs)

    def preprocess_data(self):
        """
        Main method to preprocess the dataset according to specified configurations.

        :return: Depending on the mode, returns the splitted dataframe and an exit flag.
        """
        exit = False
        try:
            print('\nData preprocessing in progress...\n')

            ########## NaN MANAGEMENT ##########
            self.df, exit = self.manage_nan(self.df)
      
            if exit:
                raise Exception('The dataset has been modified, please reload the file')
            
            ######## END NaN MANAGEMENT ########


            ########### REMOVING NON-NUMERIC COLUMNS ############
            
            # If there are columns containing non-numeric characters (excluding dates) they are removed
            #non_numeric_cols = self.df.select_dtypes(include=['object']).columns
            # Remove the target column from the list of columns to be deleted, if it is of object type
            #non_numeric_cols = non_numeric_cols.drop(self.target_column, errors='ignore')
            # Deletes the non-numeric columns from the DataFrame
            #self.df.drop(columns=non_numeric_cols, inplace=True)      
            #############################
            
            
            ############## SPLIT DATASET ##############

            train, test = self.split_data(self.df)

            #######################

            ######### OUTLIER MANAGEMENT #########
            if self.run_mode != "test":
                # Removing outliers from the training set
                train = self.replace_outliers(train)

            ######### END OUTLIER MANAGEMENT #########
            
            ############## DATA SCALING ##############
            if self.scaling:

                if self.run_mode == "train" or self.run_mode == "train_test":
                    scaler = MinMaxScaler()
                    # fit the scaler on the training set
                    train = train.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
                    scaler.fit(train[train.columns[0:train.columns.shape[0] - 1]])
                    # save training scaling data with pickle
                    with open(f"{self.folder_path}/scaler.pkl", "wb") as file:
                        pickle.dump(scaler, file)
                    # scale training data    
                    train[train.columns[0:train.columns.shape[0] - 1]] = scaler.transform(train[train.columns[0:train.columns.shape[0] - 1]])
                    if self.run_mode == "train_test":    
                        # scale test data
                        test = test.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
                        test[test.columns[0:test.columns.shape[0] - 1]] = scaler.transform(test[test.columns[0:test.columns.shape[0] - 1]])
                        

                if self.run_mode == "test":
                    # load scaling data from pkl file
                    with open(f"{self.model_path}/scaler.pkl", "rb") as file:
                        scaler = pickle.load(file)
                    # The last column is the date column, so it is not considered
                    num_features = test.columns.shape[0] - 1
                    test = test.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
                    test[test.columns[0:num_features]] = scaler.transform(test[test.columns[0:num_features]]) 
                
                if self.run_mode == "fine_tuning": 
                    # load scaling data from pkl file
                    with open(f"{self.model_path}/scaler.pkl", "rb") as file:
                        scaler = pickle.load(file)
                    num_features = train.columns.shape[0] - 1
                    train[train.columns[0:num_features]] = scaler.transform(train[train.columns[0:num_features]])
                    test[test.columns[0:num_features]] = scaler.transform(test[test.columns[0:num_features]])   

            ############ END DATA SCALING ###########

            print("Data preprocessing complete")
            if self.run_mode == "test":
                 return test, exit
            else:    
                 return train, test, exit

        except Exception as e:
            print(f"An error occurred during preprocessing: {e}")
            return None
         
    def manage_nan(self, df, max_nan_percentage=50, min_nan_percentage=10, percent_threshold = 40):
        """
        Manage NaN values in the dataset based on defined percentage thresholds and interpolation strategies.

        :param df: Dataframe to analyze
        :param max_nan_percentage: Maximum allowed percentage of NaN values for a column to be interpolated or kept
        :param min_nan_percentage:  Minimum percentage of NaN values for which linear interpolation is applied
        :param percent_threshold: Threshold percentage of NaNs in the target column to decide between interpolation and splitting the dataset
        :return: A tuple (df, exit), where df is the DataFrame after NaN management, and exit is a boolean flag indicating if the dataset needs to be split
        """
        # Save the original index
        original_index = self.df.index
        # Reset the index
        df.reset_index(drop=True, inplace=True)
        # percent_threshold is the percentage threshold of NaNs in the target column to split the file
        exit = False
        # Calculate the percentage of NaNs for each column
        nan_percentages = df.isna().mean() * 100
        # Columns for linear interpolation
        lin_interpol_cols = nan_percentages[(nan_percentages > 0) & (nan_percentages < min_nan_percentage)].index
        # Columns for polynomial interpolation
        pol_interpol_cols = nan_percentages[(nan_percentages >= min_nan_percentage) & (nan_percentages <= max_nan_percentage)].index
        # Apply linear interpolation
        df[lin_interpol_cols] = df[lin_interpol_cols].interpolate(method='linear', limit_direction='both')
        # Apply polynomial interpolation
        df[pol_interpol_cols] = df[pol_interpol_cols].interpolate(method='polynomial', order=2, limit_direction='both')
        # Columns other than target with NaN percentage higher than the maximum allowed
        columns_to_drop = nan_percentages[(nan_percentages > max_nan_percentage) & (nan_percentages.index != self.target_column)].index
        # Remove columns other than target with high percentage of NaNs
        df.drop(columns=columns_to_drop, inplace=True)
        # Operations for the target column if it has a number of NaNs above the maximum threshold 
        if nan_percentages[self.target_column] > max_nan_percentage:         
        # Calculate indices related to NaN holes in all DataFrame columns
            nan_hole = self.detect_nan_hole(df)   
            # If there is a hole in the target column
            if nan_hole[self.target_column][0] is not None:
                # Calculate the size of the NaN hole
                hole_dim = nan_hole[self.target_column][1] - nan_hole[self.target_column][0]
                # If the percentage of the NaN hole is less than a set threshold, fill NaNs with polynomial interpolation
                if hole_dim/len(df) * 100 < percent_threshold:
                    df[self.target_column].interpolate(method='polynomial', inplace=True)
                # Otherwise, split the file into two separate files and return an exit flag 'exit'
                else:
                    self.split_file_at_nanhole(nan_hole)
                    print('\nThe dataset has been divided. Restart and launch with the new dataset.\n')
                    exit = True
                    return df, exit
            # If there is no hole in the target column, fill NaNs with polynomial interpolation
            else:
                df[self.target_column].interpolate(method='polynomial', inplace=True)

        df.index = original_index
        return df, exit
        
    def detect_nan_hole(self, df):
        """
        Detects the largest contiguous NaN hole in the target column.

        :param df: DataFrame in which to find the NaN hole
        :return: A dictionary with the start and end indices of the largest NaN hole in the target column
        """
        target_column = self.target_column
        # Dictionary to store the start and end indices of the consecutive NaN group for the target column
        nan_hole = {}
        # Find the target column in the DataFrame
        target = df[target_column]
        # Find NaN values in the target column
        is_nan = target.isna()
        # Calculate groups of consecutive NaN or non-NaN values
        groups = is_nan.ne(is_nan.shift()).cumsum()
        # Select only the groups containing NaNs
        consecutive_nan_groups = groups[is_nan]
        # If there are no consecutive NaN groups, record None for start and end
        if consecutive_nan_groups.empty:
            nan_hole[target_column] = (None, None)
        else:
            # Calculate the lengths of the groups and find the longest group
            group_lengths = consecutive_nan_groups.value_counts()
            longest_group = group_lengths.idxmax()
            # Find the start and end indices of the longest consecutive NaN group
            group_start = consecutive_nan_groups[consecutive_nan_groups == longest_group].index.min()
            group_end = consecutive_nan_groups[consecutive_nan_groups == longest_group].index.max()
            # Record the start and end indices in the dictionary
            nan_hole[target_column] = (group_start, group_end)
                
        return nan_hole    
    
    # TO BE MODIFIED: HANDLE OTHER EXTENSIONS AS WELL
    def split_file_at_nanhole(self, nan_hole):
        """
        Splits the dataset at a significant NaN hole into two separate files.

        :param nan_hole: Dictionary containing start and end indices of the NaN hole in the target column
        """
        target_column = self.target_column
        # Extract the start and end indices from the target column within nan_hole
        start, end = nan_hole[target_column]
        # Save the time when you are creating the csvs
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
        # Create the name for the first CSV file (0 to the first index)    
        first_file_name = f"dataset_part_1_{timestamp}{self.file_ext}"      
        csv1 = self.df.iloc[:start+1]
        csv1.to_csv(first_file_name, index=False)
        # Create the name for the second CSV file (from the second index to the end of the group in the target column)
        second_file_name = f"dataset_part_2_{timestamp}{self.file_ext}"
        csv2 = self.df.iloc[end+1:]
        csv2.to_csv(second_file_name, index=False)
    
    def replace_outliers(self,df):
        """
        Replaces outliers in the dataset based on the Interquartile Range (IQR)
        method. Instead of analyzing the entire dataset at once, this method focuses on a window of data points at a time. 
        The window moves through the data series step by step. For each step, it includes the next data point
        in the sequence while dropping the oldest one, thus maintaining a constant
        window size. For each position of the window, the function calculates the
        first (Q1) and third (Q3) quartiles of the data within the window. These
        quartiles are used to determine the Interquartile Range (IQR), from which
        lower and upper bounds for outliers are derived.


        :param df: DataFrame from which to remove and replace outliers
        :return: DataFrame with outliers replaced
        """
        # Set the window size and k factor
        window_size = 7  # Increase if execution is slow
        k = 1.5  # standard factor for IQR
        # Select only numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        total_outliers = 0
        # Calculate IQR only for numeric columns
        for column in numeric_cols:
            # Calculate IQR only for numeric columns
            Q1 = df[column].rolling(window= window_size).quantile(0.25)
            Q3 = df[column].rolling(window= window_size).quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (k * IQR)
            upper_bound = Q3 + (k * IQR)
            
            # Count outliers
            outliers_lower = (df[column] < lower_bound).sum()
            outliers_upper = (df[column] > upper_bound).sum()
            # Add up outliers for each column
            total_outliers += outliers_lower + outliers_upper

            # Replace values below the lower limit with the lower limit itself
            df[column] = df[column].mask(df[column] < lower_bound, lower_bound)
            # Replace values above the upper limit with the upper limit itself
            df[column] = df[column].mask(df[column] > upper_bound, upper_bound)
        self.conditional_print("Number of outliers:", total_outliers)
     
        return df

    def print_stats(self, train):
        """
        Print statistics for the selected feature in the training dataset.

        :param train: DataFrame containing the training data
        """
        # Print on the standard output the statistics of the dataset (for the selected feature)
        max_value = train[self.target_column].max()
        min_value = train[self.target_column].min()
        mean_value = train[self.target_column].mean()
        variance_value = train[self.target_column].var()

        # Creating and printing a table with the statistics
        stats_train = pd.DataFrame({
            'MAX': [max_value],
            'MIN': [min_value],
            'MEAN': [mean_value],
            'VARIANCE': [variance_value],
        })
        print(f'Statistics for the target column "{self.target_column}":')
        print(stats_train)
        print('\n')

    def split_data(self, df):
        """
        Split the dataset into training and test sets.
        If a list with dates is given, each set is created within the respective dates, otherwise the sets are created following 
        the given percentage sizes.

        :param df: DataFrame to split
        :return: Tuple of DataFrames for training, testing, and validation
        """
        # Data splitting for test mode
        match self.run_mode:

            case 'test':
                # Convert into int values the list containing Int64Index elements
                self.dates = [index[0] for index in self.dates]
                test = df[self.dates[0]:self.dates[1]]
                return None, test, None
            
            case 'train':
                # Convert into int values the list containing Int64Index elements
                self.dates = [index[0] for index in self.dates]
                train = df[self.dates[0]:self.dates[1]]
                return train, None, None
            
            case 'train_test':
                # Convert into int values the list containing Int64Index elements
                self.dates = [index[0] for index in self.dates]
                train = df[self.dates[0]:self.dates[1]]
                test = df[self.dates[2]:self.dates[3]]
                return train, test
                        

        
    


    
    