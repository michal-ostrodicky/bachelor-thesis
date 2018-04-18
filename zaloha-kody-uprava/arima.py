import numpy as np
import pandas as pd
import pywt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.robust import mad
import statsmodels.api as sm
import seaborn as sns
import sys

sns.set(color_codes=True)



'''
    Funkcia na zjemnenie casoveho radu
'''
def waveletTransformation(x, wavelet="db4", level=1, title=None):
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

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

'''
    Predikcia cien pomocou ARIMA, 
    pouziva sa rolling prediction, v ktorom si predikovane hdnoty ulozim a postupne porovnavam s
    testovacimi hodnotami. 
'''
def prediction_arima(X,size):
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    t = 0

    while t < len(test):
        model = ARIMA(history, order=(3, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(24)
        predicted_value = output[0]
        # print(predicted_value)
        q = t + 24
        observation = test[t:q]
        for i in range(len(observation)):
            predictions.append(predicted_value[i])
            history.append(observation[i])
            # print('predicted=%f, expected=%f' % (predicted_value[i], observation[i]))
        t = q

    print("MAPE testovacie ", mean_absolute_percentage_error(test, predictions))

    print("BUDUCNOST")
    output = model_fit.forecast(24)
    predicted_value = output[0]
    print("NEXT VALUES: ", predicted_value)


    return model_fit


def main():
    # PREPARING DATA
    if len(sys.argv) > 1:
        # argv[1] has your filename
        filename = sys.argv[1]
        market = sys.argv[2]



    data_xls = pd.read_excel("elspot-prices_2017_hourly_eur.xls", 'elspot-prices_2017_hourly_eur', index_col=None, )
    data_xls.to_csv('prices.csv', encoding='utf-8')

    data_csv = pd.read_csv(filename)
    # data_csv = read_csv('prices.csv', engine='python', skipfooter=1)
    # dataset = data_csv.values
    # dataset = dataset.astype('float32')

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
    data = data_csv[market]

    data.fillna((data.mean()), inplace=True)

    # sns.distplot(data[market]);

    # Plots
    # data.index = data_csv['Hours'].values
    # plt.figure(figsize=(18,9))
    # plt.plot(data.index,data)
    # plt.legend(loc='upper right')
    # plt.ylabel('Price')
    # plt.xlabel('Date')
    # plt.title('Price of electricity');
    # plt.legend(loc='upper right')
    # plt.grid(which='major', axis='both', linestyle='--')
    # plt.show()
    #
    #
    # fig = plt.figure(figsize=(12,8))
    # ax1 = fig.add_subplot(211)
    # fig = sm.graphics.tsa.plot_acf(data, lags=30, ax=ax1)
    # ax2 = fig.add_subplot(212)
    # fig = sm.graphics.tsa.plot_pacf(data, lags=30, ax=ax2)
    # plt.show()

    ## ADF statistic If the value is larger than the critical values, again,
    # meaning that we can accept the null hypothesis and in turn that the time series is non-stationary

    # print("Fuller test stacionarity test: ",sm.tsa.stattools.adfuller(data[market]))

    size = round(0.6 * data.shape[0])- (round(0.6 * data.shape[0]) % 24)
    datum = data_csv['Hours'][size:len(data)]


    data= waveletTransformation(data)
    model_fit =  prediction_arima(data,size)


if __name__ == '__main__':
    main()
