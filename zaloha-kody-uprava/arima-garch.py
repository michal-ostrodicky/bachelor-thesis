import numpy as np
import pandas as pd
import pywt
import pyflux as pf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from statsmodels.robust import mad
import statsmodels.api as sm
import seaborn as sns
sns.set(color_codes=True)



def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


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

def prediction_hybrid(X,market,size,end_date):
    # size = int(len(X[market].values) * 0.66)

    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    w = pywt.Wavelet('db4')
    t = 0
    print(len(test))

    while t < len(test):
        cA, cD = pywt.dwt(history, w)
        # print(cA.shape)
        # print(cD.shape)
        model = ARIMA(cA, order=(3, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(24)
        predicted_value = output[0]
        # print(predicted_value)

        model = pf.GARCH(cD, p=1, q=0)
        x = model.fit()
        # print(x.summary())
        radCD1 = model.predict(h=24)
        pred = np.asarray(radCD1.iloc[:,0])

        arima_prediction = pywt.upcoef('a', predicted_value, 'db4', take=24)
        garch_prediction = pywt.upcoef('d', pred, 'db4',take=24)
        vysledok = arima_prediction + garch_prediction

        q = t + 24
        observation = test[t:q]

        t = q
        for i in range(len(observation)):
            predictions.append(vysledok[i])
            history.append(observation[i])
            # print('predicted=%f, expected=%f' % (vysledok[i], observation[i]))

        if(t % 25):
            print(t)
        # prediction = pywt.idwt(predicted_value, pred, w)
        # print(prediction)
    #
    #
    #     # radcA = model.predict(h=1, intervals=False)
    #     #  print(radcA)
    #
    #     # novy_cA = output.as_matrix()
    #     # novy_cA = novy_cA[:, 0]
    #
    #     # print(novy_cA.shape)
    #     # print(type(novy_cA))
    #     # print(novy_cA)
    #     # print(output)
    #     # print(type(output))
    #     novy_cA = np.asarray(output[0])
    #
    #
    #

    print("MAPE testovacie ", mean_absolute_percentage_error(test, predictions))

    print("BUDUCNOST")
    cA, cD = pywt.dwt(history, w)
    model = ARIMA(cA, order=(3, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(24)
    predicted_value = output[0]

    model = pf.GARCH(cD, p=1, q=0)
    x = model.fit()
    # print(x.summary())
    radCD1 = model.predict(h=24)
    pred = np.asarray(radCD1.iloc[:, 0])

    arima_prediction = pywt.upcoef('a', predicted_value, 'db4', take=24)
    garch_prediction = pywt.upcoef('d', pred, 'db4', take=24)
    vysledok = arima_prediction + garch_prediction

    print("NEXT VALUES: ", vysledok)

    datumy = [None] * 24
    td = np.timedelta64(1, 'h')

    last_date = pd.to_datetime(end_date)

    datumy[0] = last_date + 24 * td

    for i in range(23):
        datumy[i + 1] = datumy[i] + td

    output = np.round(vysledok, 2)
    result_model = [datumy, output]

    # print("Predikcia: ", result_network)
    for i in range(24):
        print(result_model[0][i], " ", result_model[1][i])


def main():
    # PREPARING DATA
    data_xls = pd.read_excel("elspot-prices_2018_hourly_eur.xls", 'elspot-prices_2018_hourly_eur', index_col=None)
    data_xls.to_csv('prices.csv', encoding='utf-8')

    data_csv = pd.read_csv("prices.csv")

    end_date = data_csv.iat[data_csv.shape[0] - 1, 0]
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
    market = 'Kr.sand'
    data = data_csv
    data = data[market]
    data.fillna((data.mean()), inplace=True)
    # sns.distplot(data[market]);

    # Plots
    # data.index = data_csv['Hours'].values
    # plt.figure(figsize=(18, 9))
    # plt.plot(data.index, data[market])
    # plt.legend(loc='upper right')
    # plt.ylabel('Price')
    # plt.xlabel('Date')
    # plt.title('Price of electricity');
    # plt.legend(loc='upper right')
    # plt.grid(which='major', axis='both', linestyle='--')
    # plt.show()

    # fig = plt.figure(figsize=(12, 8))
    # ax1 = fig.add_subplot(211)
    # fig = sm.graphics.tsa.plot_acf(data[market], lags=30, ax=ax1)
    # ax2 = fig.add_subplot(212)
    # fig = sm.graphics.tsa.plot_pacf(data[market], lags=30, ax=ax2)
    # plt.show()

    ## ADF statistic If the value is larger than the critical values, again,
    # meaning that we can accept the null hypothesis and in turn that the time series is non-stationary

    print("Fuller test stacionarity test: ", sm.tsa.stattools.adfuller(data))

    size = round(0.6 * data.shape[0]) - (round(0.6 * data.shape[0]) % 24)
    datum = data_csv['Hours'][size:len(data)]

    data = waveletTransformation(data)
    prediction_hybrid(data,market,size,end_date)


if __name__ == '__main__':
    main()



