from itertools import product
from tqdm import tqdm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
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

def adf_test(series, alpha=0.05, max_diff=3, verbose=True):
    """
    Esegue il test ADF (Augmented Dickey-Fuller) in modo iterativo, aumentando
    il numero di differenziazioni fino a trovare la stazionarietà o raggiungere max_diff.

    Parametri
    ----------
    series : pd.Series o pd.DataFrame
        Serie temporale da testare. Se è un DataFrame con più colonne,
        viene considerata la prima colonna .iloc[:, 0].
    alpha : float, opzionale
        Livello di significatività (default=0.05).
    max_diff : int, opzionale
        Numero massimo di differenziazioni da provare (default=3).
    verbose : bool, opzionale
        Se True, stampa i risultati di ogni passaggio (default=True).

    Ritorno
    -------
    dict
        Un dizionario con i campi:
        - 'd': il numero di differenziazioni con cui si ottiene la stazionarietà
               (oppure max_diff se non si ottiene entro tale soglia).
        - 'Test Statistic': valore ADF dell'ultimo test eseguito.
        - 'p-value': p-value dell'ultimo test.
        - 'Critical Values': i valori critici dell'ultimo test.
        - 'Stationary': booleano che indica se la serie risulta stazionaria
                        all'uscita della funzione.
    """
    # Se passiamo un DataFrame, prendiamo la prima colonna
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    # Mappa alpha -> chiave corrispondente nei valori critici di ADF (1%, 5%, 10%)
    alpha_mapping = {0.01: '1%', 0.05: '5%', 0.1: '10%'}
    crit_key = alpha_mapping.get(alpha, '5%')  # di default usiamo '5%' se non trova corrispondenza

    # Copia della serie (per differenziazioni successive)
    current_series = series.copy()

    # Risultati finali
    final_result = {
        'd': 0,
        'Test Statistic': None,
        'p-value': None,
        'Critical Values': {},
        'Stationary': False
    }

    for d in range(max_diff + 1):
        # Eseguiamo ADF sulla serie differenziata d volte
        # (se d=0, è la serie originale)
        adf_result = adfuller(current_series.dropna(), autolag='AIC')
        adf_stat, p_value, used_lag, nobs, crit_vals, icbest = adf_result

        # Prepariamo informazioni per output
        if verbose:
            print(f"Test ADF con differenziazione d={d}")
            print(f"  ADF Statistic: {adf_stat:.4f}")
            print(f"  p-value:       {p_value:.4f}")
            print(f"  Valori critici:")
            for k, v in crit_vals.items():
                print(f"    {k}: {v}")

        # Valore critico corrispondente alla soglia alpha selezionata (default 5%)
        critical_val = crit_vals.get(crit_key, crit_vals['5%'])  # fallback su '5%' se chiave non trovata

        # Criterio combinato: p-value < alpha e adf_stat < critical_val -> stazionaria
        is_stationary = (p_value < alpha) and (adf_stat < critical_val)

        # Aggiorno i risultati
        final_result['d'] = d
        final_result['Test Statistic'] = adf_stat
        final_result['p-value'] = p_value
        final_result['Critical Values'] = crit_vals
        final_result['Stationary'] = is_stationary

        if is_stationary:
            if verbose:
                print(f"\n=> d value for ADF stationarity: {d}.\n")
            break
        else:
            if d < max_diff:
                # Differenziamo nuovamente la serie per il prossimo giro
                current_series = current_series.diff()
            else:
                if verbose:
                    print(f"\n=> Non si è raggiunta la stazionarietà entro d={max_diff}.\n")

    return final_result

def kpss_test(series, alpha=0.05, regression='c'):
    """
    Esegue il test KPSS (Kwiatkowski-Phillips-Schmidt-Shin) su una serie temporale.

    Parametri
    ----------
    series : array-like, pandas Series/DataFrame
        Serie temporale su cui eseguire il test.
    alpha : float, opzionale
        Livello di significatività desiderato (default=0.05).
    regression : str, opzionale
        Specifica il tipo di 'regression' da usare nel test:
        - 'c' per stationarietà attorno a una costante (default)
        - 'ct' per stationarietà attorno a costante + trend

    Ritorno
    -------
    dict
        Un dizionario con:
        - 'Test Statistic': valore della statistica KPSS
        - 'p-value': p-value associato
        - 'n_lags': numero di lags usati
        - 'Critical Values': dizionario dei valori critici
        - 'Stationary': booleano che indica se, a livello alpha, non si respinge l'ipotesi di stazionarietà
    """
    # Esegue il test KPSS
    statistic, p_value, n_lags, critical_values = kpss(series.dropna(), regression=regression)

    # Scelta del valore critico in base ad alpha
    # Nota: i critical values del KPSS di solito sono forniti per 10%, 5%, 2.5%, 1%
    # Se alpha non è direttamente tra questi, ci si deve approssimare o fissare una soglia.
    # Qui facciamo un esempio per alpha = 0.05 e i corrispondenti critical values '5%'.
    crit_key = None
    # Cerchiamo la chiave "più vicina" a alpha in critical_values
    # (ad es. se alpha=0.05, useremo '5%'; se alpha=0.1, useremo '10%'; ecc.)
    available_levels = [float(k.strip('%'))/100 for k in critical_values.keys()]  # es. [0.1, 0.05, 0.025, 0.01]
    if alpha in available_levels:
        # Se alpha è esattamente uno dei livelli presenti
        crit_key = f"{int(alpha*100)}%"
    else:
        # Trova la chiave più vicina (in difetto o eccesso) – semplifichiamo con min() sul differenziale
        best_match = min(available_levels, key=lambda x: abs(x - alpha))
        crit_key = f"{int(best_match*100)}%"

    # Verifichiamo se "non respingiamo" l'ipotesi nulla (serie stazionaria) a livello alpha
    # Il KPSS ha ipotesi nulla: la serie è stazionaria (a differenza di ADF).
    # Se la statistica > valore critico => respinge l'ipotesi di stazionarietà
    if crit_key in critical_values:
        stationary_decision = statistic < critical_values[crit_key]
    else:
        # Se, ad esempio, alpha non è presente tra i valori critici, facciamo un check con la più vicina
        stationary_decision = statistic < critical_values[list(critical_values.keys())[0]]  # fallback

    results = {
        'Test Statistic': statistic,
        'p-value': p_value,
        'n_lags': n_lags,
        'Critical Values': critical_values,
        'Stationary': stationary_decision
    }

    return results


def pp_test(series, alpha=0.05):
    """
    Esegue il test Phillips-Perron (PP) su una serie temporale.

    Parametri
    ----------
    series : array-like, pandas Series/DataFrame
        Serie temporale su cui eseguire il test.
    alpha : float, opzionale
        Livello di significatività desiderato (default=0.05).

    Ritorno
    -------
    dict
        Un dizionario con:
        - 'Test Statistic': valore della statistica PP
        - 'p-value': p-value associato
        - 'Critical Values': dizionario dei valori critici
        - 'Stationary': booleano che indica se, a livello alpha, si respinge l'ipotesi di radice unitaria
          (quindi, se True, suggerisce stazionarietà in termini di radice unitaria).
    """
    # Esegue il test Phillips-Perron
    # Il risultato è un oggetto di tipo 'PhillipsPerronTestResults'
    # che contiene statistic, pvalue, critical_values, ...
    pp_result = phillips_perron(series.dropna())

    statistic = pp_result.stat
    p_value = pp_result.pvalue
    critical_values = pp_result.critical_values

    # Mappatura semplificata di alpha -> chiave corrispondente
    # I valori critici forniti da PP spesso sono '1%', '5%', '10%'.
    mapping = {0.01: '1%', 0.05: '5%', 0.1: '10%'}
    crit_key = mapping.get(alpha, '5%')  # default 5% se alpha non corrisponde

    # Se la statistica è minore del valore critico (ed il p-value < alpha),
    # si respinge l'ipotesi nulla di radice unitaria, quindi la serie è (secondo PP) stazionaria.
    stationary_decision = (p_value < alpha) and (statistic < critical_values[crit_key])

    results = {
        'Test Statistic': statistic,
        'p-value': p_value,
        'Critical Values': critical_values,
        'Stationary': stationary_decision
    }

    return results


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


def time_s_analysis(df, target_column, seasonal_period, d = 0, D = 0):
    """
    Performs ACF and PACF analysis on the original and differentiated time series based on provided orders of differencing.

    :param df: The DataFrame containing the time series data.
    :param target_column: The column in the DataFrame representing the time series to analyze.
    :param seasonal_period: The period to consider for seasonal decomposition and autocorrelation analysis.
    :param d: Order of non-seasonal differencing.
    :param D: Order of seasonal differencing.
    """

    ######### STATISTICAL TESTS ######

    adf_results = adf_test(series=df[target_column])
    kpss_results = kpss_test(df[target_column], alpha=0.05, regression='c')


    if adf_results['Stationary'] and kpss_results['Stationary']:
        print(f"La serie è stazionaria.")
    elif adf_results['Stationary'] and (not kpss_results['Stationary']):
        print(f"La serie è stazionaria secondo il test ADF, e non stazionaria secondo il test kpss.")
    elif (not adf_results['Stationary']) and (kpss_results['Stationary']):
        print(f"La serie è non stazionaria secondo il test ADF, e stazionaria secondo il test kpss.")
    else:
        print(f"La serie non è risultata stazionaria.")

    kpss_results_diff = kpss_test(df.diff()[target_column], alpha=0.05, regression='c')

    if kpss_results_diff['Stationary']:
        print(f"Serie differenziata stazionaria secondo il test kpss.")

    # Plot the time series
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

    """# Creazione di un grafico per ciascun giorno della settimana 

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
        plt.show()"""

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

    """# Distribution by week day and 15-minute interval of the day
    df['interval_day'] = df.index.hour * 4 + df.index.minute // 15 + 1
    mean_day_interval = df.groupby(["week_day", "interval_day"])[target_column].mean()

    mean_day_interval.plot()
    plt.title(f"Mean {target_column} During Week", fontsize=10)
    plt.xticks([i * 24 for i in range(7)], ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    plt.xlabel("Day and 15-minute interval")
    plt.ylabel(f"Number of {target_column}")
    plt.tight_layout()
    plt.show()"""

    #pp_results = pp_test(df[target_column], alpha=0.05)

    # Plot ACF and PACF for the original series
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
    plt.show()
    