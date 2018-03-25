import numpy as np
import pywt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.robust import mad



'''
    Funkcia na zjemnenie casoveho radu

'''


def wavelet_smooth(x, wavelet="db4", level=1, title=None):
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
def prediction_arima_flask(X,market,size):
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    t = 0

    while t < len(test):
        model = ARIMA(history, order=(2, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast(24)
        predicted_value = output[0]
        # print(predicted_value)
        q = t + 24
        observation = test[t:q]
        for i in range(len(observation)):
            predictions.append(predicted_value[i])
            history.append(observation[i])
        t = q

    MAPE = mean_absolute_percentage_error(test, predictions)
    print("MAPE testovacie ",MAPE)

    print("BUDUCNOST")
    output = model_fit.forecast(24)
    predicted_value = output[0]
    print("NEXT VALUES: ", predicted_value)

    return model_fit,MAPE,predicted_value


def main():
    pass



if __name__ == '__main__':
    main()

