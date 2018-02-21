from flask import Flask,render_template,request
app = Flask(__name__)
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
from os.path import join, dirname, realpath
import os
import numpy as np
import pandas as pd
import pywt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.robust import mad



UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = set(['xls', 'csv'])
global data
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')



def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#
# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'],
#                                filename)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        # check if the post request has the file part
        if 'dataset' not in request.files:
            print("Something went wrong")
            # return redirect(request.url)
        file = request.files['dataset']
        # if user does not select file, browser also
        # submit a empty part without filename

        if file.filename == '':
            return render_template('index.html' ,empty_upload=True)

        if file and allowed_file(file.filename) :
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('uploaded_file',
                      #              filename=filename))

            filename_without_extension, file_extension = os.path.splitext(filename)
            path_saved_file = os.path.join(app.config['UPLOAD_FOLDER']) + "/" + filename

            ## otvorenie file podla toho, aky extension ma file

            if(file_extension == '.xls' ):
                data_xls = pd.read_excel(path_saved_file, filename_without_extension, index_col=None)
                data_xls.to_csv('prices.csv', encoding='utf-8')
                data_csv = pd.read_csv("prices.csv")

            elif(file_extension == '.csv'):
                data_csv = pd.read_csv(path_saved_file)

            column_names = list(data_csv.columns.values)
            global data
            data = data_csv
            #print(column_names)


            return render_template('upload.html', labels = column_names)
        else:
            return render_template('index.html', invalid_extension = True)

@app.route('/choose', methods=['POST'])
def predict_label():
    if request.method == 'POST':
        myvar = request.form["label_to_predict"]
        print(myvar)
        mse = prediction_arima(data)
        return render_template('blank.html', predik = mse)


def prediction_arima(data_csv):
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

    size = int(len(data[market].values) * 0.66)
    datum = data_csv['Hours'][size:len(data[market].values)].values

    data[market] = waveletSmooth(data[market].values)
    chyba = predikuj(data[market].values,size)
    return chyba


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
def predikuj(X,size):
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()

    for t in range(len(test)):
        model = ARIMA(history, order=(1, 1, 0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        predicted_value = output[0]
        predictions.append(predicted_value)
        observation = test[t]
        history.append(observation)
        # print('predicted=%f, expected=%f' % (yhat, obs))

    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    return error

if __name__ == '__main__':
    app.run(debug = True)
