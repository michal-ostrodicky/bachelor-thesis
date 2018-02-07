import numpy as np
import pandas as pd
import pywt
import xlrd
import pyflux as pf
from arch import arch_model
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.robust import mad

# PREPARING DATA
data = xlrd.open_workbook("elspot-prices_2017_hourly_sek.xls")

data_xls = pd.read_excel('elspot-prices_2017_hourly_sek.xls', 'elspot-prices_2017_hourly_sek', index_col=None)
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
market = 'Oslo'
data = data_csv[['Hours',market]]



# Plotts
data.index = data_csv['Hours'].values
plt.figure(figsize=(18,9))
plt.plot(data.index,data[market])
plt.legend(loc='upper right')
plt.ylabel('Price')
plt.xlabel('Date')
plt.title('Price of electricity');
plt.legend(loc='upper right')
plt.grid(which='major', axis='both', linestyle='--')
plt.show()


size = int(len(data[market].values) * 0.66)
# datum = data[size:len(data['Bergen'].values)]
# datum = datum['Hours']

datum2 = data_csv['Hours'][size:len(data[market].values)].values
datum = datum2


def waveletSmooth( x, wavelet="db4", level=1, title=None ):
    # calculate the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode="per" )
    # calculate a threshold
    sigma = mad( coeff[-level] )
    # changing this threshold also changes the behavior,
    # but I have not played with this very much
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:] )
    # reconstruct the signal using the thresholded coefficients
    X = pywt.waverec( coeff, wavelet, mode="per" )

    size = int(len(X) * 0.66)

    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()


    for t in range(len(test)):
        model = ARIMA(history, order=(3, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        # print('predicted=%f, expected=%f' % (yhat, obs))

    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)

    plotting_frame = pd.DataFrame(predictions)
    plotting_frame.index = datum2


    plt.figure(figsize=(18, 9))
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.title('Price of electricity');
    plt.plot(data.index,data[market].values, label = market)
    plt.plot(plotting_frame.index,plotting_frame, color='red', label = 'Predicted')
    plt.legend(loc='upper right')
    plt.grid(which='major', axis='both', linestyle='--')

    plt.show()



waveletSmooth(data[market].values)

def plotFigure(data_plot,file_name,order):
    fig = plt.figure(order, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_plot)
    fig.savefig(file_name, bbox_inches='tight')
    plt.close()

# X = data[market].values
# print(len(X))
# size = int(len(X) * 0.66)
# print('Size je ', size)
# train, test = X[0:size], X[size:len(X)]
# history = [x for x in train]
# predictions = list()
#
# for t in range(len(test)):
#     model = ARIMA(history, order=(3, 1, 0))
#     model_fit = model.fit(disp=0)
#     output = model_fit.forecast()
#     yhat = output[0]
#     predictions.append(yhat)
#     obs = test[t]
#     history.append(obs)
#     print('predicted=%f, expected=%f' % (yhat, obs))
#
# error = mean_squared_error(test, predictions)
# print('Test MSE: %.3f' % error)


