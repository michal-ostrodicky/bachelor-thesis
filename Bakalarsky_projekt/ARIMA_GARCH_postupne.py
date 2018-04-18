import numpy as np
import pandas as pd
import pywt
import pyflux as pf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.robust import mad
import sys


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


def main():
    # PREPARING DATA

    if len(sys.argv) > 1:
        # argv[1] has your filename
        filename = sys.argv[1]
        market = sys.argv[2]

    data_csv = pd.read_csv(filename)

    end_date = data_csv.iat[data_csv.shape[0] - 1, 0]
    # Pridanie hodiny k datumu

    data = data_csv
    data = data[market]
    data.fillna((data.mean()), inplace=True)


    size = round(0.6 * data.shape[0]) - (round(0.6 * data.shape[0]) % 24)
    datum = data_csv['Hours'][size:len(data)]

    data = waveletTransformation(data)

    train, test = data[0:size], data[size:len(data)]
    history = [x for x in train]
    predictions = list()
    w = pywt.Wavelet('db4')
    t = 0

    while t < len(test):
        cA, cD = pywt.dwt(history, w)
        model = ARIMA(cA, order=(3, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(24)
        predicted_value = output[0]


        model = pf.GARCH(cD, p=1, q=0)
        x = model.fit()
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



    print("MAPE na testovacích dátach: ", mean_absolute_percentage_error(test, predictions))


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

    datumy = [None] * 24
    td = np.timedelta64(1, 'h')

    last_date = pd.to_datetime(end_date)

    datumy[0] = last_date + 24 * td

    for i in range(23):
        datumy[i + 1] = datumy[i] + td

    output = np.round(vysledok, 2)
    result_model = [datumy, output]

    print("Predikcia cien elektriny na nasledujúcich 24 hodín:")
    for i in range(24):
        print(result_model[0][i], " ", result_model[1][i])

if __name__ == '__main__':
    main()



