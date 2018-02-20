import numpy as np
import pandas as pd
import pywt
import xlrd
import pyflux as pf
from arch import arch_model
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.robust import mad
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as scs
import statsmodels.tsa.api as smt
sns.set(color_codes=True)


'''
    Funkcia na zjemnenie casoveho radu

'''
def waveletSmooth(x, wavelet="db4", level=1, title=None):
    # calculate the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet, mode="per")
    # calculate a threshold
    sigma = mad(coeff[-level])
    # changing this threshold also changes the behavior
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    # reconstruct the signal using the thresholded coefficients
    X = pywt.waverec(coeff, wavelet, mode="per")

    return X


'''
    Predikcia cien pomocou ARIMA, 
    pouziva sa rolling prediction, v ktorom si predikovane hdnoty ulozim a postupne porovnavam s
    testovacimi hodnotami. 
'''
def prediction_arima(X):
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()

    for t in range(len(test)):
        model = ARIMA(history, order=(3, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()

        predicted_value = output[0]
        predictions.append(predicted_value)
        observation = test[t]
        history.append(observation)
        # print('predicted=%f, expected=%f' % (yhat, obs))

    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)

    # plotting result
    plotting_frame = pd.DataFrame(predictions)
    plotting_frame.index = datum

    plt.figure(figsize=(18, 9))
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.title('Price of electricity');
    plt.plot(data.index, data[market].values, label=market)
    plt.plot(plotting_frame.index, plotting_frame, color='red', label='Predicted')
    plt.legend(loc='upper right')
    plt.grid(which='major', axis='both', linestyle='--')
    plt.show()

def _get_best_model(TS):
    best_aic = np.inf
    best_order = None
    best_mdl = None

    pq_rng = range(5) # [0,1,2,3,4]
    q_rng = range(5) # [0,1]
    for i in pq_rng:
        for j in pq_rng:
            try:
                tmp_mdl = ARIMA(TS, order=(i,1,j)).fit(
                    method='mle', trend='nc'
                )
                tmp_aic = tmp_mdl.aic

                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (i, 1, j)
                    best_mdl = tmp_mdl
            except: continue
    #print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))
    return best_aic, best_order, best_mdl


def tsplot(y, lags=None, figsize=(15, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        # mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        # y.index = data_csv['Hours'].values
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        # qq_ax = plt.subplot2grid(layout, (2, 0))
        # pp_ax = plt.subplot2grid(layout, (2, 1))

        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
        # sm.qqplot(y, line='s', ax=qq_ax)
        # qq_ax.set_title('QQ Plot')
        # scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
        plt.show()
    return

# PREPARING DATA
# data = xlrd.open_workbook("elspot-prices_2018_hourly_eur.xls")

data_xls = pd.read_excel('elspot-prices_2018_hourly_eur.xls', 'elspot-prices_2018_hourly_eur', index_col=None)
data_xls.to_csv('prices.csv', encoding='utf-8')


data_csv = pd.read_csv("prices.csv")

# Pridanie hodiny k datumu
dates = data_csv[data_csv.columns[0:2]]
dates.columns = ['Day', 'Hour']
dates['Hour'] = dates['Hour'].map(lambda x: str(x)[:2])

df = pd.DataFrame(dates)
df['Period'] = df.Day.astype(str).str.cat(df.Hour.astype(str), sep=' ')
df['Period'] = pd.to_datetime(df["Period"])

data_csv['Hours'] = df['Period']
data_csv = data_csv.drop(data_csv.columns[[0]], axis=1)


# VYBER STLPCA, pre ktory chceme robit predikciu
market = 'Bergen'
data = data_csv

# sns.distplot(data[market]);

# Plots
# data.index = data_csv['Hours'].values
# plt.figure(figsize=(18,9))
# plt.plot(data.index,data[market])
# plt.legend(loc='upper right')
# plt.ylabel('Price')
# plt.xlabel('Date')
# plt.title('Price of electricity');
# plt.legend(loc='upper right')
# plt.grid(which='major', axis='both', linestyle='--')
# plt.show()

#
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(data[market], lags=40, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(data[market], lags=40, ax=ax2)
# plt.show()

## ADF statistic If the value is larger than the critical values, again,
# meaning that we can accept the null hypothesis and in turn that the time series is non-stationary
# print("Fuller test stacionarity test: ",sm.tsa.stattools.adfuller(data[market]))

# tsplot(data[market],lags=30)
size = int(len(data[market].values) * 0.66)
datum = data_csv['Hours'][size:len(data[market])].values



data[market] = waveletSmooth(data[market].values)
res_tup = _get_best_model(data[market].values)
print(res_tup)
# prediction_arima(data[market].values)
# order = res_tup[1]
# model = res_tup[2]

# p_ = order[0]
# o_ = order[1]
# q_ = order[2]

# print("p_ %d o_%d q_ %d", p_,o_,q_)