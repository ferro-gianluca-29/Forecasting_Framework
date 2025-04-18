import matplotlib.pyplot as plt
from itertools import product
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import MSTL, STL, seasonal_decompose
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
import pandas as pd
import numpy as np

def conditional_print(verbose, *args, **kwargs):
    """
    Prints messages conditionally based on a verbosity flag.

    :param verbose: Boolean flag indicating whether to print messages.
    :param args: Arguments to be printed.
    :param kwargs: Keyword arguments to be printed.
    """
    if verbose:
        print(*args, **kwargs)

def adf_test(df, alpha=0.05, max_diff=3):
    """
    Esegue il test di Dickey-Fuller (ADF) per valutare se una serie è stazionaria
    e restituisce il numero di differenziazioni necessarie.
    Stampa inoltre i dettagli dei risultati ad ogni step.

    Parametri
    ----------
    df : pandas.DataFrame o pandas.Series
        La serie temporale da analizzare. Se è un DataFrame, si considera la prima colonna.
    alpha : float, opzionale
        Il livello di significatività desiderato (default=0.05).
    max_diff : int, opzionale
        Numero massimo di differenziazioni da provare (default=3).

    Ritorno
    -------
    int
        Il numero di differenziazioni richiesto per rendere stazionaria la serie
        (se entro max_diff). Se non viene raggiunta la stazionarietà entro max_diff,
        restituisce max_diff.
    """
    # Se df è un DataFrame multi-colonna, consideriamo la prima colonna
    if isinstance(df, pd.DataFrame):
        df = df.iloc[:, 0]

    print("Test di stazionarietà in corso...\n")

    # Mappa alpha al corrispondente valore critico
    critical_mapping = {0.01: '1%', 0.05: '5%', 0.1: '10%'}
    crit_key = critical_mapping.get(alpha, '5%')

    d = 0  # contatore di differenziazioni

    # -- Test sulla serie originale (d=0)
    adf_result = adfuller(df.dropna())
    adf_stat, p_value = adf_result[0], adf_result[1]
    critical_val = adf_result[4][crit_key]

    print(f"Test su serie originale (d={d}):")
    print(f"  ADF Statistic: {adf_stat:.4f}")
    print(f"  p-value:       {p_value:.10f}")
    print("  Valori Critici:")
    for k, v in adf_result[4].items():
        print(f"    {k}: {v}")

    # Verifica di stazionarietà
    if p_value < alpha and adf_stat < critical_val:
        print("\nLa serie è già stazionaria (d=0).")
        return d

    # -- Itera con differenze successive fino a max_diff
    df_diff = df.copy()
    for i in range(1, max_diff + 1):
        d = i
        df_diff = df_diff.diff()  # differenza cumulativa

        adf_result = adfuller(df_diff.dropna())
        adf_stat, p_value = adf_result[0], adf_result[1]
        critical_val = adf_result[4][crit_key]

        print(f"\nTest su serie differenziata (d={d}):")
        print(f"  ADF Statistic: {adf_stat:.4f}")
        print(f"  p-value:       {p_value:.4f}")
        print("  Valori Critici:")
        for k, v in adf_result[4].items():
            print(f"    {k}: {v}")

        if p_value < alpha and adf_stat < critical_val:
            print(f"\nLa serie è risultata stazionaria con d={d}.")
            return d

    print(f"\nNon è stata raggiunta la stazionarietà entro d={max_diff}.")
    return max_diff


def ljung_box_test(residuals):
        """
        Conducts the Ljung-Box test on the residuals of a fitted time series model to check for autocorrelation.

        :param model: The time series model after fitting to the data.
        """
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        lb_pvalue = lb_test['lb_pvalue'].iloc[0]
        if lb_pvalue > 0.05:
            print('Ljung-Box test result:\nNull hypothesis valid: Residuals are uncorrelated\n')
        else:
            print('Ljung-Box test result:\nNull hypothesis invalid: Residuals are correlated\n')

def multiple_STL(dataframe,target_column):
    """
    Performs multiple seasonal decomposition using STL on specified periods.

    :param dataframe: The DataFrame containing the time series data.
    :param target_column: The column in the DataFrame to be decomposed.
    """
    
    mstl = MSTL(dataframe[target_column], periods=[24, 24 * 7, 24 * 7 * 4])
    res = mstl.fit()

    fig, ax = plt.subplots(nrows=2, figsize=[10,10])
    res.seasonal["seasonal_24"].iloc[:24*3].plot(ax=ax[0])
    ax[0].set_ylabel(target_column)
    ax[0].set_title("Daily seasonality")

    res.seasonal["seasonal_168"].iloc[:24*7*3].plot(ax=ax[1])
    ax[1].set_ylabel(target_column)
    ax[1].set_title("Weekly seasonality")

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(nrows=2, figsize=[10,10])
    res.seasonal["seasonal_168"].iloc[:24*7*3].plot(ax=ax[0])
    ax[0].set_ylabel(target_column)
    ax[0].set_title("Weekly seasonality")

    res.seasonal["seasonal_672"].iloc[:24*7*4*3].plot(ax=ax[1])
    ax[1].set_ylabel(target_column)
    ax[1].set_title("Monthly seasonality")

    plt.tight_layout()
    plt.show()


def prepare_seasonal_sets(train, valid, test, target_column, period):
    """
    Decomposes the datasets into seasonal and residual components based on the specified period.

    :param train: Training dataset.
    :param valid: Validation dataset.
    :param test: Test dataset.
    :param target_column: The target column in the datasets.
    :param period: The period for seasonal decomposition.
    :return: Decomposed training, validation, and test datasets.
    """
    
    # Seasonal and residual components of the training set
    train_seasonal = pd.DataFrame(seasonal_decompose(train[target_column], model='additive', period=period).seasonal) 
    train_seasonal.rename(columns = {'seasonal': target_column}, inplace = True)
    train_seasonal = train_seasonal.dropna()
    train_residual = pd.DataFrame(seasonal_decompose(train[target_column], model='additive', period=period).resid)
    train_residual.rename(columns = {'resid': target_column}, inplace = True)
    train_residual = train_residual.dropna()
    # Seasonal and residual components of the validation set
    valid_seasonal = pd.DataFrame(seasonal_decompose(valid[target_column], model='additive', period=period).seasonal)
    valid_seasonal.rename(columns = {'seasonal': target_column}, inplace = True)
    valid_seasonal = valid_seasonal.dropna()
    valid_residual = pd.DataFrame(seasonal_decompose(valid[target_column], model='additive', period=period).resid)
    valid_residual.rename(columns = {'resid': target_column}, inplace = True)
    valid_residual = valid_residual.dropna()
    # Seasonal and residual components of the test set
    test_seasonal = pd.DataFrame(seasonal_decompose(test[target_column], model='additive', period=period).seasonal)
    test_seasonal.rename(columns = {'seasonal': target_column}, inplace = True)
    test_seasonal = test_seasonal.dropna()
    test_residual = pd.DataFrame(seasonal_decompose(test[target_column], model='additive', period=period).resid)
    test_residual.rename(columns = {'resid': target_column}, inplace = True)
    test_residual = test_residual.dropna()

    # Merge residual and seasonal components on indices with 'inner' join to keep only matching rows
    train_merge = pd.merge(train_residual, train_seasonal, left_index=True, right_index=True, how='inner')
    valid_merge = pd.merge(valid_residual, valid_seasonal, left_index=True, right_index=True, how='inner')
    test_merge = pd.merge(test_residual, test_seasonal, left_index=True, right_index=True, how='inner')
    
    # Add the residual and seasonal columns
    train_decomposed = pd.DataFrame(train_merge.iloc[:,0] + train_merge.iloc[:,1])
    train_decomposed = train_decomposed.rename(columns = {train_decomposed.columns[0]: target_column})
    valid_decomposed = pd.DataFrame(valid_merge.iloc[:,0] + valid_merge.iloc[:,1])
    valid_decomposed = valid_decomposed.rename(columns = {valid_decomposed.columns[0]: target_column})
    test_decomposed = pd.DataFrame(test_merge.iloc[:,0] + test_merge.iloc[:,1])
    test_decomposed = test_decomposed.rename(columns = {test_decomposed.columns[0]: target_column})

    return train_decomposed, valid_decomposed, test_decomposed

def time_s_analysis(df, target_column, seasonal_period, d = 0, D = 0):
    """
    Performs ACF and PACF analysis on the original and differentiated time series based on provided orders of differencing.

    :param df: The DataFrame containing the time series data.
    :param target_column: The column in the DataFrame representing the time series to analyze.
    :param seasonal_period: The period to consider for seasonal decomposition and autocorrelation analysis.
    :param d: Order of non-seasonal differencing.
    :param D: Order of seasonal differencing.
    """

    """# Plot the time series
    df = df.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
    df[target_column] = df[target_column].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
    plt.plot(df['date'], df[target_column], 'b')
    plt.title('Time Series')
    plt.xlabel('Time series index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Plot the time series 1 settimana
    df = df.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
    df[target_column] = df[target_column].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
    plt.plot(df['date'].iloc[:24 * 7], df[target_column].iloc[:24 * 7], 'b')
    plt.title('Time Series 1 settimana')
    plt.xlabel('Time series index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    #     # Plot the time series 1 giorno
    df = df.applymap(lambda x: x.replace(',', '.') if isinstance(x, str) else x)
    df[target_column] = df[target_column].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
    plt.plot(df['date'].iloc[:24], df[target_column].iloc[:24], 'b')
    plt.title('Time Series 1 giorno')
    plt.xlabel('Time series index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    df['weekday'] = df.index.day_name()
    df['minutes'] = df.index.hour * 60 + df.index.minute
    df['hours'] = df.index.hour + df.index.minute / 60

    # Creazione di un grafico per ciascun giorno della settimana

    week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for day in week_days:
        plt.figure(figsize=(14, 10))
        daily_data = df[df['weekday'] == day]
        for label, grp in daily_data.groupby(daily_data.index.date):
            plt.plot(grp['hours'], grp[target_column], label=label, alpha=0.5)
        plt.title(day)
        plt.xlabel('Hours from midnight')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Distribution by month
    df['month'] = df.index.month
    df.boxplot(column=target_column, by='month', flierprops={'markersize': 3, 'alpha': 0.3})
    df.groupby('month')[target_column].median().plot(style='o-', linewidth=0.8)
    plt.xlabel('Month')
    plt.title(f'{target_column} Distribution by Month', fontsize=9)
    plt.show()

    # Distribution by week day
    df['week_day'] = df.index.day_of_week + 1
    df.boxplot(column=target_column, by='week_day', flierprops={'markersize': 3, 'alpha': 0.3})
    df.groupby('week_day')[target_column].median().plot(style='o-', linewidth=0.8)
    plt.ylabel(target_column)
    plt.title(f'{target_column} Distribution by Week Day', fontsize=9)
    plt.show()

    # Distribution by hour of the day
    df['hour_of_day'] = df.index.hour
    df.boxplot(column=target_column, by='hour_of_day', flierprops={'markersize': 3, 'alpha': 0.3})
    df.groupby('hour_of_day')[target_column].median().plot(style='o-', linewidth=0.8)
    plt.ylabel(target_column)
    plt.title(f'{target_column} Distribution by Hour of the Day', fontsize=9)
    plt.show()

    # Distribution by week day and 15-minute interval of the day
    df['interval_day'] = df.index.hour * 4 + df.index.minute // 15 + 1
    mean_day_interval = df.groupby(["week_day", "interval_day"])[target_column].mean()

    mean_day_interval.plot()
    plt.title(f"Mean {target_column} During Week", fontsize=10)
    plt.xticks([i * 24 for i in range(7)], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    plt.xlabel("Day and 15-minute interval")
    plt.ylabel(f"Number of {target_column}")
    plt.tight_layout()
    plt.show()"""

    adf_d = adf_test(df=df[target_column])
    print(f"Suggested d from Dickey-Fuller Test: {adf_d}")


    
    """# Plot ACF and PACF for the original series
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_acf(df[target_column], lags=seasonal_period + 4, ax=axes[0, 0])
    axes[0, 0].set_title('ACF of Original Series')
    axes[0, 0].set_xlabel('Lags')
    axes[0, 0].set_ylabel('Autocorrelation')

    plot_pacf(df[target_column], lags=seasonal_period + 4, ax=axes[0, 1])
    axes[0, 1].set_title('PACF of Original Series')
    axes[0, 1].set_xlabel('Lags')
    axes[0, 1].set_ylabel('Partial Autocorrelation')
    
    # Applying non-seasonal differencing
    differenced_series = df[target_column].copy()
    for _ in range(d):
        differenced_series = differenced_series.diff().dropna()

    # Applying seasonal differencing
    for _ in range(D):
        differenced_series = differenced_series.diff(seasonal_period).dropna()

    # Ensure data cleaning after differencing
    differenced_series.dropna(inplace=True)
    
    # ACF and PACF plots for the differentiated series
    plot_acf(differenced_series, lags=seasonal_period + 4, ax=axes[1, 0])
    axes[1, 0].set_title(f'ACF of Differenced Series (d = {d}, D = {D})')
    axes[1, 0].set_xlabel('Lags')
    axes[1, 0].set_ylabel('Autocorrelation')

    plot_pacf(differenced_series, lags=seasonal_period + 4, ax=axes[1, 1])
    axes[1, 1].set_title(f'PACF of Differenced Series (d = {d}, D = {D})')
    axes[1, 1].set_xlabel('Lags')
    axes[1, 1].set_ylabel('Partial Autocorrelation')

    plt.tight_layout()
    plt.show()

    # Time series decomposition into its trend, seasonality, and residuals components
    decomposition = STL(df[target_column][:seasonal_period*30], period=seasonal_period).fit()

    # Personalizza il plot per mostrare solo i dati di 30 giorni
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(10, 8), sharex=True)

    # Serie temporale originale in blu
    ax1.plot(decomposition.observed, color='blue')
    ax1.set_title('Energy Consumption Time Series', fontsize=12)
    ax1.set_ylabel('Value (MW)', fontsize=10, rotation=0, labelpad=30)  # Aggiunge spazio tra l'etichetta e l'asse
    ax1.tick_params(axis='y', labelsize=8)  # Riduce il font dei valori delle ordinate


    # Componente di trend in verde
    ax2.plot(decomposition.trend, color='green')
    ax2.set_title('Trend Component', fontsize=12)
    ax2.set_ylabel('Value (MW)', fontsize=10, rotation=0, labelpad=30)  # Aggiunge spazio tra l'etichetta e l'asse
    ax2.tick_params(axis='y', labelsize=8)  # Riduce il font dei valori delle ordinate


    # Componente stagionale in rosso
    ax3.plot(decomposition.seasonal, color='red')
    ax3.set_title('Seasonal Component', fontsize=12)
    ax3.set_ylabel('Value (MW)', fontsize=10, rotation=0, labelpad=30)  # Aggiunge spazio tra l'etichetta e l'asse
    ax3.tick_params(axis='y', labelsize=8)  # Riduce il font dei valori delle ordinate


    # Componente residua in nero
    ax4.plot(decomposition.resid, color='black')
    ax4.set_title('Residual Component', fontsize=12)
    ax4.set_ylabel('Value (MW)', fontsize=10, rotation=0, labelpad=30)  # Aggiunge spazio tra l'etichetta e l'asse
    ax4.tick_params(axis='y', labelsize=8)  # Riduce il font dei valori delle ordinate


    # Label dell'ascissa solo sull'ultimo subplot
    ax4.set_xlabel('Date', fontsize=10)

    fig.autofmt_xdate()  # Auto-format the date labels of the x-axis
    plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding
    plt.show()

    # Time series decomposition into its trend, seasonality, and residuals components
    decomposition = STL(differenced_series[:seasonal_period*30], period=seasonal_period).fit()
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,8))

    ax1.plot(decomposition.observed)
    ax1.set_ylabel('Observed')

    ax2.plot(decomposition.trend)
    ax2.set_ylabel('Trend')

    ax3.plot(decomposition.seasonal)
    ax3.set_ylabel('Seasonal')

    ax4.plot(decomposition.resid)
    ax4.set_ylabel('Residuals')

    fig.autofmt_xdate()
    plt.tight_layout()
    # Add title
    plt.suptitle(f"Time Series Decomposition of differenced series with period {seasonal_period}")
    plt.show()"""
    