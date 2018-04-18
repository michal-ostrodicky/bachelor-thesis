import numpy as np
import pandas as pd
import pywt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.robust import mad
import sys

'''
    Funkcia na zjemnenie casoveho radu
'''


def waveletTransformation(series, wavelet="db4", level=1):

    coeffs = pywt.wavedec(series, wavelet, mode="per")
    sigma = mad(coeffs[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(series)))

    coeffs[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeffs[1:])
    new_series = pywt.waverec(coeffs, wavelet, mode="per")

    return new_series


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

'''
    Predikcia cien pomocou ARIMA, 
    pouziva sa rolling prediction, v ktorom si predikovane hdnoty ulozim a postupne porovnavam s
    testovacimi hodnotami. 
'''


def main():
    if len(sys.argv) > 1:
        # argv[1] has your filename
        filename = sys.argv[1]
        market = sys.argv[2]

    data_csv = pd.read_csv(filename)
    end_date = data_csv.iat[data_csv.shape[0] - 1, 0]

    # VYBER STLPCA, pre ktory chceme robit predikciu
    data = data_csv[market]

    data.fillna((data.mean()), inplace=True)

    size = round(0.6 * data.shape[0])- (round(0.6 * data.shape[0]) % 24)

    data= waveletTransformation(data)

    train, test = data[0:size], data[size:len(data)]
    history = [x for x in train]
    predictions = list()
    t = 0

    while t < len(test):
        model = ARIMA(history, order=(3, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(24)
        predicted_value = output[0]

        q = t + 24
        observation = test[t:q]
        for i in range(len(observation)):
            predictions.append(predicted_value[i])
            history.append(observation[i])
        t = q

    print("MAPE na testovacích dátach: ", mean_absolute_percentage_error(test, predictions))

    model = ARIMA(history, order=(3, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(24)
    predicted_value = output[0]

    datumy = [None] * 24
    td = np.timedelta64(1, 'h')

    last_date = pd.to_datetime(end_date)

    datumy[0] = last_date + 24 * td

    for i in range(23):
        datumy[i + 1] = datumy[i] + td

    output = np.round(predicted_value, 2)
    result_arima = [datumy, output]

    print("Predikcia cien elektriny na nasledujúcich 24 hodín:")
    for i in range(24):
        print(result_arima[0][i], " ", result_arima[1][i])



if __name__ == '__main__':
    main()
