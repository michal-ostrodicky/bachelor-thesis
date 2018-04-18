import numpy as np
import pandas as pd
import pywt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.robust import mad
import sys
from matplotlib import pyplot



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
def prediction_arima(series,size,end_date):
    train, test = series[0:size], series[size:len(series)]
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

    print("MAPE na testovacich datach: ", mean_absolute_percentage_error(test, predictions))

    pyplot.figure(figsize=(18, 9))
    pyplot.plot(test, 'b')
    pyplot.plot(predictions, 'r')

    pyplot.legend(['Skutočné', 'Predikované'])
    pyplot.ylabel('Cena')
    pyplot.title('Cena elektriny - testovacia sada');
    pyplot.show()

    print("BUDUCNOST")
    model = ARIMA(history, order=(3, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast(24)
    predicted_value = output[0]
    print("NEXT VALUES: ", predicted_value)

    datumy = [None] * 24
    td = np.timedelta64(1, 'h')


    last_date = pd.to_datetime(end_date)

    datumy[0] = last_date + 24 * td

    for i in range(23):
        datumy[i + 1] = datumy[i] + td

    output = np.round(predicted_value, 2)
    result_network = [datumy, output]

    # print("Predikcia: ", result_network)
    for i in range(24):
        print(result_network[0][i], " ", result_network[1][i])

    pyplot.figure(figsize=(18, 9))

    pyplot.plot(result_network[0], result_network[1], 'y')

    pyplot.legend(['Budúca cena'])
    pyplot.ylabel('Cena')
    pyplot.title('Cena elektriny');
    pyplot.show()

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
    model_fit =  prediction_arima(data,size,end_date)




if __name__ == '__main__':
    main()
